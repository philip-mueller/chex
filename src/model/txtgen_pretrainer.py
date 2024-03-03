
from dataclasses import dataclass, field
from typing import Any, List, Optional
import einops

import numpy as np
from omegaconf import MISSING
import torch
import torch.nn as nn
from metrics.textgen_metrics import NLGMetrics
from model import txt_decoder, txt_encoder
from util.model_utils import BaseModel, BaseModelOutput, MainModelConfig, instantiate_model
from util.prompt_utils import flatten_prompts
from util.train_utils import EvalConfig, Evaluator


@dataclass
class TxtGenPreTrainerOutput(BaseModelOutput):
    sentences: List[List[str]] = field(default_factory=list)

@dataclass
class TxtGenPreTrainerConfig(MainModelConfig):
    txt_encoder: Any = MISSING
    txt_decoder: Any = MISSING
    enc_dec_dropout: float = 0.0
    copy_token_min: int = 1
    copy_token_max: int = 1
    max_sentences: Optional[int] = None

class TxtGenPreTrainer(BaseModel):
    CONFIG_CLS = TxtGenPreTrainerConfig

    def __init__(self, config: TxtGenPreTrainerConfig, from_checkpoint=False) -> None:
        self.config: TxtGenPreTrainerConfig
        super().__init__(config)
        self.txt_encoder = instantiate_model(
            self.config.txt_encoder, model_module=txt_encoder, main_config=self.config)
        self.txt_decoder = instantiate_model(
                self.config.txt_decoder, model_module=txt_decoder, main_config=self.config)
        
        assert self.config.copy_token_min >= 1, "copy_token_min must be >= 1"
        assert self.config.copy_token_min <= self.config.copy_token_max, "copy_token_min must be <= copy_token_max"
        self.dropout = nn.Dropout(self.config.enc_dec_dropout)
        
    def forward(self, sentences: List[Optional[List[str]]], epoch: int=None, generate=False, generation_kwargs=None, **kwargs) -> TxtGenPreTrainerOutput:
        if self.config.max_sentences is not None and self.training:
            total_sent = 0
            for i, sample_sents in enumerate(sentences):
                    remaining_sents = self.config.max_sentences - total_sent
                    if remaining_sents == len(sample_sents):
                        sentences = sentences[:i+1]
                        break
                    elif remaining_sents < len(sample_sents):
                        sentences[i] = sample_sents[:remaining_sents]
                        sentences = sentences[:i+1]
                        break
                    else:
                        total_sent += len(sample_sents)

        flattened_sentences, sentence_mask = flatten_prompts(sentences, device='cuda')
        # (N*S x d)
        flattened_features = self.txt_encoder.encode_sentences(flattened_sentences, cache=True, epoch=epoch)
        R: int = np.random.randint(self.config.copy_token_min, self.config.copy_token_max+1)
        flattened_features = einops.repeat(flattened_features, 'n d -> n r d', r=R)
        flattened_features = self.dropout(flattened_features)

        if generate:
            generation_kwargs = generation_kwargs or {}
            flat_generated_sentences = self.txt_decoder.generate(flattened_features, **generation_kwargs)
            generated_sentences = []
            i_gen = 0
            for n_sent in [len(sents) for sents in sentences]:
                if n_sent is None:
                    generated_sentences.append(None)
                else:
                    generated_sentences.append(flat_generated_sentences[i_gen:i_gen+n_sent])
                    i_gen += n_sent
            return TxtGenPreTrainerOutput(sentences=generated_sentences)
        else:
            loss = self.txt_decoder.train_step(flattened_features, flattened_sentences)
            return TxtGenPreTrainerOutput(loss=loss, step_metrics={'txtgen_loss': loss.detach()})
        
    def build_evaluator(self, task: EvalConfig, **kwargs) -> Evaluator:
        if task.task is None:
            return TxtGenPreTrainEvaluator(task, self, **kwargs)
        
@dataclass
class TxtGenPreTrainEvalConfig(EvalConfig):
    generation_kwargs: dict = field(default_factory=dict)

    
class TxtGenPreTrainEvaluator(Evaluator):
    def __init__(self, task: TxtGenPreTrainEvalConfig, model: TxtGenPreTrainer, **kwargs):
        super().__init__(task, TxtGenPreTrainEvalConfig, **kwargs)
        self.model = model
        self._register_metric(NLGMetrics(use_bleu=True, use_meteor=True, use_rouge=True, micro=True))

    def eval_step(self, sentences, **kwargs) -> TxtGenPreTrainerOutput:
        output: TxtGenPreTrainerOutput = self.model(sentences=sentences, **kwargs, generate=True, generation_kwargs=self.config.generation_kwargs)

        flate_preds = [pred for preds in output.sentences for pred in preds]
        flat_targets = [sent for sents in sentences for sent in sents]
        self._update_metric(preds=flate_preds, targets=flat_targets)
        return output
    