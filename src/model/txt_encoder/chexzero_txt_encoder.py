

from dataclasses import dataclass
import os
from typing import List
from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from model.components.mlp import MLP

from util.model_utils import BaseModel, BaseModelConfig, MainModelConfig
import chexzero.clip
 
@dataclass
class ChexzeroTextEncoderConfig(BaseModelConfig):
    model_path: str = os.path.expanduser("~/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt")
    frozen_language_model: bool = True

    # Additional projection layers (after the language model projection)
    # 0 = no projection, 1 = linear, 2 = one hidden layer
    n_projection_layers: int = 0
    # whether to use batch norm in the addtional projection layers
    projection_bn: bool = False
    normalize_projected: bool = False


class ChexzeroTextEncoder(BaseModel):
    CONFIG_CLS = ChexzeroTextEncoderConfig
    MODIFYABLE_CONFIGS = ('frozen_backbone', )

    def __init__(self, config: ChexzeroTextEncoderConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: ChexzeroTextEncoderConfig

        self.d = main_config.d_model

        model, _ = chexzero.clip.load("ViT-B/32", device='cpu', jit=False) 
        model.load_state_dict(torch.load(self.config.model_path, map_location='cpu'))
        self.d = main_config.d_model

        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

        if self.config.frozen_language_model:
            for param in self.parameters():
                param.requires_grad = False

        d_backbone = self.text_projection.shape[1]
        
        self.projection = MLP(
            self.config.n_projection_layers, 
            d_in=d_backbone, 
            d_out=self.d, 
            use_bn=self.config.projection_bn,
            act=main_config.act,
            dropout=main_config.dropout)

        self.cached_sentence_embeddings = {}

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    @property
    def device(self):
        return self.token_embedding.weight.device

    def forward(self, 
        input_ids: torch.Tensor,
        project: bool = True,
        **kwargs) -> Tensor:

        input_ids = input_ids.to(device=self.device)

        # Encode image using backbone
        with torch.set_grad_enabled(not self.config.frozen_language_model):
            x = self.token_embedding(input_ids).type(self.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # (N_sentences x d)
            features = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)] @ self.text_projection

        if project:
            features = self.projection(features)
        return features

    
    def encode_sentences(self, sentences: List[str], cache=False, **kwargs) -> Tensor:
        if cache and self.config.frozen_language_model and all(s in self.cached_sentence_embeddings for s in sentences):
            features = torch.stack([self.cached_sentence_embeddings[s] for s in sentences], dim=0)
        else:
            input_ids = chexzero.clip.tokenize(sentences, context_length=77) 
            features = self(input_ids, project=False, **kwargs)

            if cache and self.config.frozen_language_model:
                for s, f in zip(sentences, features):
                    self.cached_sentence_embeddings[s] = f.detach().float()
        features = features.to(dtype=self.dtype)
        projected_features = self.projection(features)
        return projected_features
        