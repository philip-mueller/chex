
from dataclasses import dataclass
import logging
import os
from typing import Optional
import einops
from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from chexzero.model import VisualTransformer
from model.components.mlp import MLP
from model.img_encoder import ImageEncoderOutput

from util.model_utils import BaseModel, BaseModelConfig, MainModelConfig
import chexzero.clip

log = logging.getLogger(__name__)

@dataclass
class ChexzeroImageEncoderConfig(BaseModelConfig):
    model_path: str = os.path.expanduser("~/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt")
    frozen_backbone: bool = False
    # freeze over full training, i.e. never unfreeze
    freeze_backbone_layers: Optional[int] = None

    add_cls_features: bool = False
    use_pretrained_projection: bool = True
    # 0 = no additionl projection, 1 = linear, 2 = one hidden layer
    additional_projection_layers: int = 0
    projection_bn: bool = False
    normalize_projected: bool = False

class ChexzeroImageEncoder(BaseModel):
    CONFIG_CLS = ChexzeroImageEncoderConfig
    MODIFYABLE_CONFIGS = ('frozen_backbone', )

    def __init__(self, config: ChexzeroImageEncoderConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: ChexzeroImageEncoderConfig

        model, _ = chexzero.clip.load("ViT-B/32", device='cpu', jit=False) 
        model.load_state_dict(torch.load(os.path.expanduser(self.config.model_path), map_location='cpu'))
        self.d = main_config.d_model

        self.backbone: VisualTransformer = model.visual
        d_backbone = self.backbone.output_dim if self.config.use_pretrained_projection else self.backbone.proj.shape[0]
        self.n_layers = len(self.backbone.transformer.resblocks) + 2
        if config.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            log.info('Freezing backbone for the whole training')
            self.n_frozen_layers = self.n_layers
            self.n_currently_frozen_layers = self.n_layers
        elif config.freeze_backbone_layers is not None and config.freeze_backbone_layers != 0:
            n_transformer_layers = len(self.backbone.transformer.resblocks)
            if config.freeze_backbone_layers < 0:
                config.freeze_backbone_layers = (n_transformer_layers + 2) + config.freeze_backbone_layers
            self.freeze_layers(config.freeze_backbone_layers)
            self.n_frozen_layers = config.freeze_backbone_layers
            self.n_currently_frozen_layers = config.freeze_backbone_layers
            log.info(f'Freezing {self.n_frozen_layers}/{self.n_layers} layers of backbone for the whole training')

        self.patch_projection = MLP(
            self.config.additional_projection_layers, 
            d_in=d_backbone, 
            d_out=self.d, 
            use_bn=self.config.projection_bn,
            act=main_config.act,
            dropout=main_config.dropout)

    def freeze_layers(self, n_frozen_layers: int):
        # first layer (embedding layer)
        emb_layer_requires_grad = n_frozen_layers == 0
        for param in self.backbone.conv1.parameters():
            param.requires_grad = emb_layer_requires_grad
        self.backbone.class_embedding.requires_grad = emb_layer_requires_grad
        self.backbone.positional_embedding.requires_grad = emb_layer_requires_grad
        self.backbone.ln_pre.requires_grad = emb_layer_requires_grad

        # transformer layers
        n_transformer_layers = len(self.backbone.transformer.resblocks)
        n_frozen_transformer_layers = n_frozen_layers - 1
        for i, resblock in enumerate(self.backbone.transformer.resblocks):
            layer_requires_grad = i >= n_frozen_transformer_layers
            for param in resblock.parameters():
                param.requires_grad = layer_requires_grad

        # last layer (projection layer)
        proj_layer_requires_grad = n_frozen_layers > n_transformer_layers + 1
        for param in self.backbone.ln_post.parameters():
            param.requires_grad = proj_layer_requires_grad
        self.backbone.proj.requires_grad = proj_layer_requires_grad

    def forward(self, 
        x: Tensor, 
        **kwargs) -> ImageEncoderOutput:
        
        if x.ndim == 3:
            x = einops.repeat(x, 'n h w -> n c h w', c=3)
        device = x.device
        dtype = x.dtype
        N, _, H, W = x.shape
        assert H == W, "Only square images are supported"
        input_resolution = H
        assert self.backbone.input_resolution == input_resolution, f"Input resolution of backbone ({self.backbone.input_resolution}) does not match input resolution of image ({input_resolution})"

        # Encode image using backbone
        with torch.set_grad_enabled(not self.config.frozen_backbone):
            x = self.backbone.conv1(x)  # shape = [*, width, grid, grid]
            N, _, H, W = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            pos_emb = self.backbone.positional_embedding.to(x.dtype)  # shape = [grid ** 2 + 1, width]
            x = x + pos_emb
            x = self.backbone.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.backbone.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            # (N x d)
            cls_features = self.backbone.ln_post(x[:, 0, :])
            if self.config.add_cls_features:
                patch_features = self.backbone.ln_post(x[:, 1:, :] + x[:, 0, :].unsqueeze(1))
            else:
                # (N x HW x d)
                patch_features = self.backbone.ln_post(x[:, 1:, :])

            if self.config.use_pretrained_projection:
                cls_features = cls_features @ self.backbone.proj
                patch_features = patch_features @ self.backbone.proj
                pos_embeddings = pos_emb[1:] @ self.backbone.proj
            else:
                pos_embeddings = pos_emb[1:]

            # (N, H, W, d_backbone)
            patch_features = einops.rearrange(patch_features, 'n (h w) d -> n h w d', h=H, w=W)
            # (N, H, W, d_backbone) -> (N, H, W, d)
            pos_embeddings = einops.repeat(pos_embeddings, '(h w) d -> n h w d', h=H, w=W, n=N)
            
        # (N x H x W x d)
        projected_patch_features = self.patch_projection(patch_features)
        cls_features = self.patch_projection(cls_features.unsqueeze(1)).squeeze(1)
        if self.config.additional_projection_layers > 0:
            pos_embeddings = self.patch_projection(pos_embeddings)

        if self.config.normalize_projected:
            projected_patch_features = F.normalize(projected_patch_features, dim=-1)
            cls_features = F.normalize(cls_features, dim=-1)

        return ImageEncoderOutput(
            patch_features=projected_patch_features,
            pos_embeddings=pos_embeddings,
            global_features=cls_features)
