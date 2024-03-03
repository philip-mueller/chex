

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import einops
from omegaconf import MISSING
from torch import BoolTensor, FloatTensor, Tensor
from torch import nn
import torch
import torch.nn.functional as F
from model.components.mlp import MLP
from model.components.soft_roi_pool import SoftRoiPool
from model.components.transformer import AttentionMask, TransformerDecoderBlock, TransformerEncoderBlock, TransformerTokenDecoder

from model.img_encoder import ImageEncoderOutput
from util.model_utils import BaseModel, BaseModelConfig, BaseModelOutput, MainModelConfig
from torchvision.ops import roi_pool, box_convert


@dataclass
class TokenDetectorOutput(BaseModelOutput):
    IGNORE_APPLY = ('encoded_image', *BaseModelOutput.IGNORE_APPLY)

    # (N x Q x d) 
    box_features: FloatTensor = MISSING
    
    # (N x Q x 4)
    # in the (x_c, y_c, w, h) format (center-format)
    # with values in [0, 1] relative to the original (masked) image size
    boxes: Optional[FloatTensor] = None
    box_mask: Optional[BoolTensor] = None
    boxes_weights: Optional[FloatTensor] = None
    boxes_present: Optional[BoolTensor] = None

    # (N x Q x R x ...)
    multiboxes_features: Optional[FloatTensor] = None
    multiboxes: Optional[FloatTensor] = None
    multiboxes_weights: Optional[FloatTensor] = None
    multiboxes_present: Optional[BoolTensor] = None

    encoded_image: Optional[ImageEncoderOutput] = None


@dataclass
class TokenDetectorConfig(BaseModelConfig):
    # --- Multi-region ---
    multiregions: int = 3
    multiregion_sa: bool = True
    
    # --- Decoder layers ---
    n_decoder_layers: int = 1
    enc_dec_droppath: bool = False
    decoder_sa: bool = False
    decoder_ff: bool = True
    shortcut_tokens: bool = False
    use_pos_embeddings: bool = True
    shortcut_pos_embeddings: bool = False
    
    # --- Bbox predictor and ROI pooling ---
    clip_pos: bool = False
    predictor_hidden_layers: int = 1
    patch_project_layers: int = 1
    out_project_layers: int = 2
    soft_roi_beta: float = 2.0
    skip_con_roi_pool: bool = True
    dropout_skip_roi_pool: float = 0.3 
    skip_roi_pool_prob: float = 0.25

class TokenDetector(BaseModel):

    CONFIG_CLS = TokenDetectorConfig
    def __init__(self, config: TokenDetectorConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: TokenDetectorConfig
        self.main_config: MainModelConfig = main_config
        self.d = main_config.d_model
        
        self.has_multiregions = self.config.multiregions > 1
        self.multiregion_tokens = nn.Parameter(torch.randn(self.config.multiregions, self.d) / self.d**0.5)
        self.multiregion_weight_mlp = MLP(
            n_layers=self.config.predictor_hidden_layers + 1, d_in=self.d, d_hidden=self.d, d_out=1,
            dropout_last_layer=False, act=main_config.act, dropout=main_config.dropout)
        
        # ----- Decoder -----
        self.decoder = TransformerTokenDecoder(
            d_model=self.d, nhead=main_config.n_head, 
            n_decoder_layers=self.config.n_decoder_layers,
            act=main_config.act, dropout=main_config.dropout, attention_dropout=main_config.attention_dropout,
            droppath_prob=main_config.droppath_prob,
            layer_scale=main_config.layer_scale, layer_scale_init=main_config.layer_scale_init,
            enc_dec_droppath=self.config.enc_dec_droppath, 
            decoder_sa=self.config.decoder_sa or self.config.multiregion_sa, 
            decoder_ff=self.config.decoder_ff,
            shortcut_tokens=self.config.shortcut_tokens, shortcut_pos_embeddings=self.config.shortcut_pos_embeddings
        )

        # ----- Bbox predictor -----
        self.bbox_predictor = nn.Sequential(
            nn.LayerNorm(self.d),
            MLP(
                n_layers=self.config.predictor_hidden_layers + 1,
                d_in=self.d, d_hidden=self.d, d_out=4,
                dropout_last_layer=False,
                act=main_config.act, dropout=main_config.dropout)
        )
        
        # ----- ROI pooling -----
        self.patch_projector = nn.Sequential(
                nn.LayerNorm(self.d),
                MLP(
                    n_layers=self.config.patch_project_layers,
                    d_in=self.d, act=main_config.act, dropout=main_config.dropout)
            )
        if self.config.skip_con_roi_pool:
            self.skip_con_dropout = nn.Dropout(self.config.dropout_skip_roi_pool)
        self.out_projector = nn.Sequential(
                    nn.LayerNorm(self.d),
                    MLP(
                        n_layers=self.config.out_project_layers,
                        d_in=self.d, act=main_config.act, dropout=main_config.dropout)
                )
        
        self.roi_pool = SoftRoiPool()
        self.beta = self.config.soft_roi_beta
        
    def forward(self, 
        encoded_image: ImageEncoderOutput,
        query_tokens: Tensor, 
        query_mask: Optional[BoolTensor] = None,
        skip_roi_pool: Optional[bool] = None
        ) -> TokenDetectorOutput:
        """
        """
        # (N x Q x d), (N x Q x H x W)
        query_features, query_mask = self.decode_queries(encoded_image, query_tokens, query_mask)
        # (N x Q x 4), (N x Q) or (1) or float
        box_params = self.predict_boxes(query_features)

        # (N x Q x d)
        box_features = self.apply_roi_pool(
            encoded_image, 
            box_params=box_params, 
            query_features=query_features, 
            beta=self.beta, 
            skip_roi_pool=skip_roi_pool)
       
        if query_mask is not None:
            box_features = box_features.masked_fill(~query_mask[..., None], 0)
            box_params = box_params.masked_fill(~query_mask[..., None], 0)
            
        output = TokenDetectorOutput(
            encoded_image=encoded_image,
            box_features=box_features,
            boxes=box_params,
            box_mask=query_mask,
        )
        output = self.handle_multiregions(output)

        return output

    def decode_queries(self, 
        encoded_image: ImageEncoderOutput,
        query_tokens: Tensor,
        query_mask: Optional[BoolTensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Apply transformer decoder to get image-based features for each query token
        :param encoded_image: ImageEncoderOutput
        :param query_tokens: (N x Q x d) or (Q x d)
        :param query_mask: (N x Q)
        :return (query_features, query_pmask)
            query_features: (N x Q x d)
        """
        assert query_tokens.shape[-1] == self.d, f"Query tokens ({query_tokens.shape[-1]}) must match model dim ({self.d})"
        assert encoded_image.patch_features.shape[-1] == self.d, f"Patch features ({encoded_image.patch_features.shape[-1]}) must match model dim ({self.d})"
        N, H, W, d = encoded_image.patch_features.shape
        # (N x (H*W) x d)
        flattened_features = encoded_image.patch_features.flatten(1, 2)
        assert query_tokens.ndim in [2, 3], f"Query tokens must be (Q x d) or (N x Q x d), got {query_tokens.shape}"
        if query_tokens.ndim == 2:
            query_tokens = einops.repeat(query_tokens, 'q d -> n q d', n=N)
        else:
            assert query_tokens.shape[0] == N, f"Query tokens ({query_tokens.shape[0]}) and patch features ({flattened_features.shape[0]}) must have the same batch size"
        N, Q, d = query_tokens.shape
        assert Q > 0, "Query tokens must have at least one token"

        if self.config.use_pos_embeddings:
            # (N x (H*W) x d_model)
            flattened_pos_emb = encoded_image.pos_embeddings.flatten(1, 2)
        else:
            flattened_pos_emb = None

        # Multiregion handling
        query_sa_mask = None
        Q_all = Q
        R = self.config.multiregions
        Q_all = Q * R

        # (N x Q x R x d)
        query_tokens = query_tokens[:, :, None, :] + self.multiregion_tokens[None, None, :, :]
        # (N x Q*R x d)
        query_tokens = query_tokens.flatten(1, 2)
        query_mask = einops.repeat(query_mask, 'n q -> n (q r)', r=R) if query_mask is not None else None

        if self.config.multiregion_sa and not self.config.decoder_sa:
            # ((Q*R) x (Q*R)) -> block diagonal matrix with QxQ blocks of size RxR
            query_sa_mask = torch.block_diag(*[torch.ones(R, R, dtype=torch.bool, device=query_tokens.device) for _ in range(Q)])
            assert query_sa_mask.shape == (Q_all, Q_all), f"Query self-attention mask must be (Q*R x Q*R), got {query_sa_mask.shape}"

        # Apply decoder
        # (N x Q' x d)
        query_features, *_ = self.decoder(
            token_features=query_tokens, token_mask=query_mask, token_sa_mask=query_sa_mask,
            region_features=flattened_features, region_pos_embeddings=flattened_pos_emb
        )

        return query_features, query_mask

    def predict_boxes(self, query_features: Tensor) -> Tuple[Tensor, Union[Tensor, float]]:
        """
        :param query_features: (N x Q x d)
        :return 
            box_params: (N x Q x 4) in format (x_c, y_c, w, h) in [0, 1] relative to image size
        """
        box_params = self.bbox_predictor(query_features)
        # (N x Q x 4)
        box_params = box_params.sigmoid()
        # (N x Q x 2)
        pos, size = box_params[..., :2], box_params[..., 2:]

        if self.config.clip_pos:
            pos = pos.clamp(0., 1.)
       
        box_params = torch.cat([pos, size], dim=-1)
        return box_params
        
    def apply_roi_pool(self, 
        encoded_image: ImageEncoderOutput,
        box_params, beta, query_features,
        skip_roi_pool: Optional[bool],
        ):
        
        patch_features = self.patch_projector(encoded_image.patch_features)
        box_features, query_patch_map = self.roi_pool(patch_features, box_params, beta=beta)
        
        if skip_roi_pool:
            box_features = (query_features + box_features) / 2
        elif skip_roi_pool is None and self.config.skip_con_roi_pool and query_features is not None:
            skip_roi_pool_mask = torch.rand(box_features.shape[:2], device=box_features.device) < self.config.skip_roi_pool_prob
            skip_features = self.skip_con_dropout(query_features)
            skip_features = (skip_features + box_features) / 2
            box_features = torch.where(skip_roi_pool_mask[..., None], skip_features, box_features)
        box_features = self.out_projector(box_features)
        
        return box_features
    
    def encode_regions(self, encoded_image, region_boxes, region_mask, region_prompt_emb=None, beta=None):
        if beta is None:
            beta = self.config.soft_roi_beta

        if region_prompt_emb is not None:
            N, Q, d = region_prompt_emb.shape
            # (N x Q x d), (N x Q x H x W)
            query_features, _, _ = self.decode_queries(encoded_image, region_prompt_emb, region_mask)
            query_features = einops.rearrange(query_features, 'n (q r) d -> n q r d', q=Q)
            query_features = query_features.mean(dim=2)
            skip_roi_pool = True
        else:
            query_features = None
            skip_roi_pool = False
        
        box_features = self.apply_roi_pool(encoded_image, region_boxes, query_features=query_features, beta=beta, skip_roi_pool=skip_roi_pool)

        if region_mask is not None:
            box_features = box_features * region_mask[:, :, None]
        return box_features
    
    def handle_multiregions(self, output: TokenDetectorOutput):
        Q_all = output.boxes.shape[1]
        N = output.boxes.shape[0]
        R = self.config.multiregions
        Q = Q_all // R
        d = output.box_features.shape[-1]
        multiboxes = einops.rearrange(output.boxes, 'n (q r) b -> n q r b', q=Q)
        multibox_features = einops.rearrange(output.box_features, 'n (q r) d -> n q r d', q=Q)
        query_mask = einops.rearrange(output.box_mask, 'n (q r) -> n q r', q=Q) if output.box_mask is not None else torch.ones((N, Q, R), dtype=torch.bool, device=multiboxes.device)

        # features weighting and aggregation
       # (N x Q x R)
        weights_logits = self.multiregion_weight_mlp(multibox_features).squeeze(-1)
        weights = weights_logits.sigmoid()
        present = weights > 0.5
        
        output.multiboxes_features = multibox_features
        output.multiboxes = multiboxes
        output.multiboxes_weights = query_mask * weights
        output.multiboxes_present = query_mask * present

        output.box_mask = query_mask.any(dim=-1)
        output.boxes_present = output.multiboxes_present.any(dim=-1)
        normed_weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1)
        output.box_features = (normed_weights[..., None] * multibox_features).sum(dim=-2)
        output.boxes_weights = (normed_weights * weights).sum(dim=-1)
        
        # use the bbox covering all multi regions
        multiboxes_upper_left_masked = (multiboxes[..., :2] - multiboxes[..., 2:] / 2) # .masked_fill(~present[..., None], 1.0)
        multiboxes_lower_right_masked = (multiboxes[..., :2] + multiboxes[..., 2:] / 2) # .masked_fill(~present[..., None], 0.0)
        boxes = torch.cat([multiboxes_upper_left_masked.min(dim=-2)[0], multiboxes_lower_right_masked.max(dim=-2)[0]], dim=-1)
        # convert back
        wh = boxes[..., 2:] - boxes[..., :2]
        boxes = torch.cat([boxes[..., :2] + wh / 2, wh], dim=-1)
        output.boxes = output.boxes_present[..., None] * boxes

        return output
        

        
class FeatureRoiPool(nn.Module):
    def forward(self, patch_features, query_boxes):
        """
        patch_features: (N x H x W x d)
        query_features: (N x Q x d)
        query_boxes: (N x Q x 4)
        query_box_masks: (N x Q)
        """
        N, H, W, d = patch_features.shape
        Q = query_boxes.shape[1]
        assert H == W
        # (N x d x H x W)
        patch_features = einops.rearrange(patch_features, "N H W d -> N d H W")
        # (N)
        box_sample_indices = torch.arange(N, device=patch_features.device, dtype=patch_features.dtype)
        # (N x Q)
        box_sample_indices = einops.repeat(box_sample_indices, "N -> N Q", Q=Q)
        # (N x Q x 5)
        boxes = torch.cat([box_sample_indices.unsqueeze(-1), query_boxes], dim=-1)
        # (K x 5)
        boxes = boxes.flatten(0, 1)  # boxes[query_box_masks]
        boxes[:, 1:] = box_convert(boxes[:, 1:], "cxcywh", "xyxy")
        # (K x d x 1 x 1)
        roi_pooled_features = roi_pool(patch_features, boxes, output_size=(1, 1), spatial_scale=H)
        roi_pooled_features = roi_pooled_features.squeeze(-1).squeeze(-1)
        # (N x Q x d)
        roi_pooled_features_reshaped = einops.rearrange(roi_pooled_features, "(N Q) d -> N Q d", N=N, Q=Q)
        #roi_pooled_features_reshaped = torch.zeros((N, Q, d), dtype=patch_features.dtype,  device=patch_features.device,)
        #roi_pooled_features_reshaped[query_box_masks] = roi_pooled_features.to(dtype=patch_features.dtype)

        
        return roi_pooled_features_reshaped

class PostDecoder(nn.Module):
    def __init__(self, n_post_decoder_layers, attend_to_patches: bool, attend_to_cls: bool, enc_dec_droppath: bool, multibox_drop_prob: float, gate: bool, main_config: MainModelConfig):
        super().__init__()

        self.attend_to_patches = attend_to_patches
        self.attend_to_cls = attend_to_cls
        self.multibox_drop_prob = multibox_drop_prob

        if attend_to_patches or attend_to_cls:
            self.post_decoder_layers = nn.ModuleList([
                    TransformerDecoderBlock(
                        d_model=main_config.d_model, nhead=main_config.n_head,
                        act=main_config.act,
                        attention_dropout=main_config.attention_dropout, dropout=main_config.dropout, droppath_prob=main_config.droppath_prob,
                        layer_scale=main_config.layer_scale, layer_scale_init=main_config.layer_scale_init,
                        encoder_droppath=enc_dec_droppath, self_attention=True, feedforward=True)
                    for _ in range(n_post_decoder_layers)])
        else:
            self.post_decoder_layers = nn.ModuleList([
                    TransformerEncoderBlock(
                        d_model=main_config.d_model, nhead=main_config.n_head,
                        act=main_config.act,
                        attention_dropout=main_config.attention_dropout, dropout=main_config.dropout, droppath_prob=main_config.droppath_prob,
                        layer_scale=main_config.layer_scale, layer_scale_init=main_config.layer_scale_init)
                    for _ in range(n_post_decoder_layers)])
        
        self.has_gate = gate
        if gate:
            self.gate = nn.Parameter(torch.zeros(1))
            self.ln_gate = nn.LayerNorm(main_config.d_model)
            
    def forward(self, 
        encoded_image: ImageEncoderOutput,
        box_features: Tensor,
        multiboxes_features: Optional[Tensor], 
        query_mask: Optional[BoolTensor] = None):
        """
        :param encoded_image: ImageEncoderOutput
        :param box_features: (N x Q x d)
        :param multiboxes_features: (N x Q x R x d)
        :param query_mask: (N x Q)
        :return (box_features, multiboxes_features)
        """
        if multiboxes_features is not None:
            N, Q, R, d = multiboxes_features.shape
            # (N x Q X R+1 x d)
            box_tokens = torch.cat([box_features.unsqueeze(2), multiboxes_features], dim=2)
            box_tokens = einops.rearrange(box_tokens, 'n q r d -> n (q r) d')

             
            # (N x Q x R+1)
            mask = einops.repeat(query_mask, 'n q -> n (q r)', r=R+1) if query_mask is not None else torch.ones((N, (Q * (R+1))), dtype=torch.bool, device=box_tokens.device)
            if self.multibox_drop_prob > 0:
                # (N x Q x R)
                keep_tokens = torch.rand((N, Q, R), device=box_tokens.device) > self.multibox_drop_prob
                # (N x Q x R+1)
                keep_tokens = torch.cat([torch.ones((N, Q, 1), dtype=torch.bool, device=box_tokens.device), keep_tokens], dim=-1)
                keep_tokens = einops.rearrange(keep_tokens, 'n q r -> n (q r)')
                mask = mask & keep_tokens
            mask = AttentionMask.create(mask, dtype=box_tokens.dtype)

            # ((Q*R+1) x (Q*R+1))
            sa_mask = torch.block_diag(*[torch.ones(R+1, R+1, dtype=torch.bool, device=box_tokens.device) for _ in range(Q)])
            sa_mask = AttentionMask.create(sa_mask, dtype=box_tokens.dtype)
        else:
            N, Q, d = box_features.shape
            box_tokens = box_features
            mask = AttentionMask.create(query_mask, dtype=box_tokens.dtype)
            sa_mask = torch.diag(torch.ones(Q, dtype=torch.bool, device=box_tokens.device))
            sa_mask = AttentionMask.create(sa_mask, dtype=box_tokens.dtype)

        source_features = None
        if self.attend_to_patches:
            source_features = encoded_image.patch_features + encoded_image.pos_embeddings
            source_features = einops.rearrange(source_features, 'n h w d -> n (h w) d')
        if self.attend_to_cls:
            # (N x 1 x d)
            cls_features = encoded_image.global_features.unsqueeze(1)
            source_features = torch.cat([source_features, cls_features], dim=1) if source_features is not None else cls_features

        if source_features is None:
            for layer in self.post_decoder_layers:
                box_tokens = layer(box_tokens, mask=mask, sa_mask=sa_mask)
        else:
            for layer in self.post_decoder_layers:
                box_tokens = layer(box_tokens, source_features=source_features, 
                                      mask=mask, source_mask=None, sa_mask=sa_mask,
                                      return_weights=False)
                
        # (N x Q x R+1 x d)
        box_tokens = einops.rearrange(box_tokens, 'n (q r) d -> n q r d', q=Q)

        if self.has_gate:
            box_features = self.ln_gate(box_features + self.gate * box_tokens[:, :, 0])
            multiboxes_features = self.ln_gate(multiboxes_features + self.gate * box_tokens[:, :, 1:]) if multiboxes_features is not None else None
        else:
            box_features = box_tokens[:, :, 0]
            multiboxes_features = box_tokens[:, :, 1:] if multiboxes_features is not None else None

        return box_features, multiboxes_features
