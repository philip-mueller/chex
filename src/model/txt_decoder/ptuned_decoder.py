
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import einops
from torch import nn
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig,  GenerationMixin, GPT2TokenizerFast, GPT2LMHeadModel, GenerationConfig
from model.components.mlp import MLP

from util.model_utils import BaseModel, BaseModelConfig, MainModelConfig


@dataclass
class PTunedDecoderConfig(BaseModelConfig):
    language_model_url: str = "healx/gpt-2-pubmed-medium"
    frozen_language_model: bool = False

    prefix_length_factor: int = 5
    n_projection_layers: int = 2
    shared_projection: bool = False

    max_length: int = 128
    # None => start at 0, 'one' => start at 1, 'prefix' => start at prefix length
    position_offset: Optional[str] = 'prefix'

    generation_kwargs: dict = field(default_factory=dict)


class PTunedDecoderModel(BaseModel):
    CONFIG_CLS = PTunedDecoderConfig
 
    def __init__(self, config: PTunedDecoderConfig, main_config: MainModelConfig):
        super().__init__(config)
        if self.config.language_model_url == "healx/gpt-2-pubmed-medium":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.config.language_model_url, max_length=self.config.max_length)
            self.language_model = GPT2LMHeadModel.from_pretrained(self.config.language_model_url)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_url, max_length=self.config.max_length)
            self.language_model = AutoModelForCausalLM.from_pretrained(self.config.language_model_url)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("Set pad token to eos token", self.tokenizer.pad_token_id)
        self.base_model_prepare_inputs_for_generation = self.language_model.prepare_inputs_for_generation
        if not isinstance(self.language_model, GenerationMixin):
            assert not hasattr(self.language_model, "generate"), "Language model already has a generate method"
            # add the generation mixin to the language model
            self.language_model.__class__ = type(f"Generation{self.language_model.__class__.__name__}", (self.language_model.__class__, GenerationMixin), {})

        lm_config: PretrainedConfig = self.language_model.config
        self.n_layers = lm_config.n_layer if hasattr(lm_config, "n_layer") else lm_config.num_hidden_layers
        self.num_heads = lm_config.n_head if hasattr(lm_config, "n_head") else lm_config.num_attention_heads
        self.d_lm = lm_config.n_embd if hasattr(lm_config, "n_embd") else lm_config.hidden_size
        if self.config.frozen_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
            self.lm_frozen = True
        else:
            self.lm_frozen = False

        self.prefix_proj = MLP(
            n_layers=self.config.n_projection_layers,
            d_in=main_config.d_model, 
            d_hidden=4 * main_config.d_model, 
            d_out=self.config.prefix_length_factor * 2 * self.n_layers * self.d_lm if not self.config.shared_projection else self.config.prefix_length_factor * 2 * self.d_lm,
            act=main_config.act, dropout=main_config.dropout
        )

    def _get_pos_offset(self, L_prefix: int):
        if self.config.position_offset is None:
            return 0
        elif self.config.position_offset == 'one':
            return 1
        elif self.config.position_offset == 'prefix':
            return L_prefix
        else:
            raise ValueError(f"Unknown position offset {self.config.position_offset}")

    def project_prefix_features(self, prefix_features) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        :param prefix_features: (N x d) or (N x L x d) for multi region
        """
        if prefix_features.ndim == 2:
            # (N x d) -> (N x 1 x d)
            prefix_features = prefix_features.unsqueeze(1)
        # (N x L x (factor_L x n_layers * 2 * n_heads * d_head))
        prefix_features = self.prefix_proj(prefix_features)
        if self.config.shared_projection:
            prefix_features = einops.repeat(
                prefix_features, "n l (fl np d) -> n l (fl nl np d)", 
                nl=self.n_layers, np=2, d=self.d_lm)
        N, L, _ = prefix_features.shape

        # (n_layers x 2 x N x n_heads x (L * prefix_L_factor) x d_head)
        prefix_features = einops.rearrange(
            prefix_features, "n l (fl nl np nh dh) -> nl np n nh (l fl) dh", 
            nl=self.n_layers, np=2, nh=self.num_heads, dh=self.d_lm // self.num_heads, n=N, l=L, fl=self.config.prefix_length_factor)
        prefix_features = tuple(
            (prefix_features[i, 0], prefix_features[i, 1]) for i in range(self.n_layers)
        )
        return prefix_features

    def train_step(self, prefix_features, target_sentences: List[str], target_sentences_mask: Optional[torch.BoolTensor] = None, **kwargs) -> torch.Tensor:
        prefix_features = self.project_prefix_features(prefix_features)
        N, _, L_prefix = prefix_features[0][0].shape[:3]
        #assert len(target_sentences) == N
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        target_sentences = [bos_token + s + eos_token for s in target_sentences]
        inputs = self.tokenizer(
                target_sentences,
                add_special_tokens=False,
                padding='longest', 
                max_length=self.config.max_length,
                truncation=True,
                return_tensors="pt")
        inputs['past_key_values'] = prefix_features
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if target_sentences_mask is not None:
            input_ids = input_ids[target_sentences_mask]
            attention_mask = attention_mask[target_sentences_mask]
            assert input_ids.shape[0] == attention_mask.shape[0] == N, f"{input_ids.shape} != {attention_mask.shape} != ({N}, ...)"

        inputs['labels'] = input_ids
        inputs['input_ids'] = input_ids
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        inputs['position_ids'] = position_ids + self._get_pos_offset(L_prefix)
        inputs['attention_mask'] = torch.cat([attention_mask.new_ones((N, L_prefix)), attention_mask], dim=1)

        outputs = self.language_model(**inputs, return_dict=True)
        
        return outputs.loss

    @torch.inference_mode()
    def generate(self, prefix_features, **kwargs):
        N = prefix_features.shape[0]
        device = prefix_features.device
        prefix_features = self.project_prefix_features(prefix_features)

        self.language_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        default_padding_side = self.tokenizer.padding_side
        #self.tokenizer.padding_side = 'left'

        generation_kwargs = dict(self.config.generation_kwargs, **kwargs)
        input_ids = self.tokenizer.bos_token_id * torch.ones((N, 1), dtype=torch.long, device=device)
        outputs = self.language_model.generate(inputs=input_ids, prefix_key_values=prefix_features, pad_token_id=self.tokenizer.pad_token_id, **generation_kwargs)
        sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.language_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        self.tokenizer.padding_side = default_padding_side

        return sentences

    def prepare_inputs_for_generation(self, *args, prefix_key_values, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        input_ids = model_kwargs["input_ids"]
        attention_mask = model_kwargs["attention_mask"]
        position_ids = model_kwargs["position_ids"]
        past_key_values = model_kwargs["past_key_values"]

        N, n_head, L_prefix, d = prefix_key_values[0][0].shape
        N_input = input_ids.shape[0]
        if N_input != N:
            assert N_input % N == 0, f"N_input does not match, {N_input} is not a multiply of {N} {(N, n_head, L_prefix, d)}"
            # repeat prefix_key_values for each input
            beam_size = N_input // N
            prefix_key_values = tuple(
                (prefix_key_values[i][0].repeat_interleave(beam_size, dim=0), prefix_key_values[i][1].repeat_interleave(beam_size, dim=0))
                for i in range(self.n_layers)
            )
            N = N_input

        if past_key_values is None:
            # ------------------ first iteration / or iterations without past_key_values ------------------ #
            # there is no past_key_values, so use prefix_key_values as only past_key_values
            model_kwargs['past_key_values'] = prefix_key_values
        # extend attention_mask to cover prefix
        model_kwargs['attention_mask'] = torch.cat([attention_mask.new_ones((N, L_prefix)), attention_mask], dim=1)
        model_kwargs['position_ids'] = position_ids + self._get_pos_offset(L_prefix)
        return model_kwargs

