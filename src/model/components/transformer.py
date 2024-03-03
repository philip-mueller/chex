
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import einops
from torch import BoolTensor, Tensor, nn
import torch

from model.components.mlp import MLP

from timm.models.layers import DropPath

from util.data_utils import TensorDataclassMixin


# ========================= Utilities and Basic Blocks =====================================


def add_pos_embeddings(features: Tensor, pos_embeddings: Optional[Tensor] = None):
    return features if pos_embeddings is None else features + pos_embeddings


def concat_masks(mask_a, mask_b, M_a, M_b, dtype):
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        N, _ = mask_b.binary_mask.shape
        bin_mask_a = mask_b.binary_mask.new_ones((N, M_a))
    else:
        bin_mask_a = mask_a.binary_mask
    if mask_b is None:
        N, _ = mask_a.binary_mask.shape
        bin_mask_b = mask_a.binary_mask.new_ones((N, M_b))
    else:
        bin_mask_b = mask_b.binary_mask

    bin_mask = torch.cat([bin_mask_a, bin_mask_b], dim=1)
    return AttentionMask.from_binary_mask(bin_mask, dtype=dtype)


def concat_pos_embeddings(pos_a, pos_b, M_a, M_b):
    if pos_a is None and pos_b is None:
        return None
    if pos_a is None:
        N, _, d = pos_b.shape
        pos_a = pos_b.new_zeros((N, M_a, d))
    if pos_b is None:
        N, _, d = pos_a.shape
        pos_b = pos_a.new_zeros((N, M_b, d))
    return torch.cat([pos_a, pos_b], dim=1)


@dataclass
class AttentionMask(TensorDataclassMixin):
    binary_mask: torch.Tensor
    inverted_binary_mask: torch.Tensor
    additive_mask: torch.Tensor

    @staticmethod
    def from_binary_mask(binary_mask: torch.Tensor, dtype):
        if binary_mask is not None:
            binary_mask = binary_mask.bool()
        additive_mask = AttentionMask._compute_additive_attention_mask(binary_mask, dtype)
        return AttentionMask(binary_mask, ~binary_mask, additive_mask)

    @staticmethod
    def create(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype=torch.float):
        if mask is None or isinstance(mask, AttentionMask):
            return mask
        else:
            assert isinstance(mask, torch.Tensor) and (mask.dtype in (torch.bool, torch.uint8, torch.int64)), \
                (type(mask), mask.dtype)
            return AttentionMask.from_binary_mask(mask, dtype)

    @staticmethod
    def _compute_additive_attention_mask(binary_attention_mask: torch.Tensor, dtype):
        if binary_attention_mask is None:
            return None
        additive_attention_mask = torch.zeros_like(binary_attention_mask, dtype=dtype)
        additive_attention_mask.masked_fill_(~binary_attention_mask, float('-inf'))
        return additive_attention_mask

    @staticmethod
    def get_binary_mask(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype=torch.float):
        if mask is None:
            return None
        else:
            return AttentionMask.create(mask, dtype=dtype).binary_mask

    @staticmethod
    def get_inverted_binary_mask(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype=torch.float):
        if mask is None:
            return None
        else:
            return AttentionMask.create(mask, dtype=dtype).inverted_binary_mask

    @staticmethod
    def get_additive_mask(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype=torch.float):
        if mask is None:
            return None
        else:
            return AttentionMask.create(mask, dtype=dtype).additive_mask


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, attention_dropout=0.0, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.sa = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout, batch_first=True)

    def forward(self, x: Tensor, mask: Optional[AttentionMask] = None, attention_mask: Optional[AttentionMask] = None):
        mask = AttentionMask.get_inverted_binary_mask(mask)

        attention_mask = AttentionMask.get_inverted_binary_mask(attention_mask)
        if attention_mask is not None and mask is not None:
            attention_mask = AttentionMask.get_inverted_binary_mask(attention_mask)
            attention_mask = einops.repeat(attention_mask, 'k l -> (n h) k l', n=x.shape[0], h=self.sa.num_heads)
            mask = einops.repeat(mask, 'n l -> (n h) k l', h=self.sa.num_heads, k=x.shape[1])
            attention_mask = attention_mask * mask
            mask = None
        
        x = self.sa(x, x, x, 
                    key_padding_mask=mask, 
                    attn_mask=attention_mask,
                    need_weights=False)[0]
        return self.dropout(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, attention_dropout=0.0, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout, batch_first=True)

    def forward(self, x: Tensor, source: Tensor, source_mask: Optional[AttentionMask] = None, return_weights=False):
        x, weights = self.mha(x, source, source, key_padding_mask=AttentionMask.get_inverted_binary_mask(source_mask), need_weights=return_weights)
        x = self.dropout(x)

        return x, weights


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        nhead: int, 
        act=nn.ReLU,
        attention_dropout=0.0, dropout=0.1, droppath_prob=0.2,
        layer_scale: bool = True, layer_scale_init=0.1):
        super().__init__()

        self.norm_sa = nn.LayerNorm(d_model)
        self.sa_block = SelfAttentionBlock(d_model, nhead, dropout=dropout, attention_dropout=attention_dropout)
        self.ls_sa = LayerScale(d_model, init_values=layer_scale_init) if layer_scale else nn.Identity()
        self.droppath_sa = DropPath(droppath_prob) if droppath_prob > 0.0 else nn.Identity()
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff_block = MLP(2, d_model, dropout=dropout, act=act)
        self.ls_ff = LayerScale(d_model, init_values=layer_scale_init) if layer_scale else nn.Identity()
        self.droppath_ff = DropPath(droppath_prob) if droppath_prob > 0.0 else nn.Identity()

    def forward(self, x, mask: Optional[AttentionMask]=None, pos_embeddings: Optional[Tensor] = None, sa_mask: Optional[AttentionMask]=None):
        x = x + self.droppath_sa(self.ls_sa(self.sa_block(self.norm_sa(add_pos_embeddings(x, pos_embeddings)), mask, attention_mask=sa_mask)))
        x = x + self.droppath_ff(self.ls_ff(self.ff_block(self.norm_ff(x))))

        return x


# ========================= Encoder / Decoder Blocks =====================================

class TransformerDecoderBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        nhead: int, 
        act=nn.ReLU,
        attention_dropout=0.0, dropout=0.1, droppath_prob=0.2, encoder_droppath: bool = False,
        layer_scale: bool = True, layer_scale_init=0.1,
        self_attention: bool = True, feedforward: bool = True):
        super().__init__()
        self.self_attention = self_attention
        if self_attention:
            self.norm_sa = nn.LayerNorm(d_model)
            self.sa_block = SelfAttentionBlock(d_model, nhead, dropout=dropout, attention_dropout=attention_dropout)
            self.ls_sa = LayerScale(d_model, init_values=layer_scale_init) if layer_scale else nn.Identity()
            self.droppath_sa = DropPath(droppath_prob) if droppath_prob > 0.0 else nn.Identity()

        self.norm_mha = nn.LayerNorm(d_model)
        self.norm_encoder = nn.LayerNorm(d_model)
        self.mha_block = CrossAttentionBlock(d_model, nhead, dropout=dropout, attention_dropout=attention_dropout)
        self.ls_mha = LayerScale(d_model, init_values=layer_scale_init) if layer_scale else nn.Identity()
        self.droppath_mha = DropPath(droppath_prob) if encoder_droppath and droppath_prob > 0.0 else nn.Identity()

        self.feedforward = feedforward
        if feedforward:
            self.norm_ff = nn.LayerNorm(d_model)
            self.ff_block = MLP(2, d_model, dropout=dropout, act=act)
            self.ls_ff = LayerScale(d_model, init_values=layer_scale_init) if layer_scale else nn.Identity()
            self.droppath_ff = DropPath(droppath_prob) if droppath_prob > 0.0 else nn.Identity()

    def forward(self, 
        x, source_features, 
        mask: Optional[AttentionMask]=None, source_mask: Optional[AttentionMask]=None, sa_mask: Optional[AttentionMask]=None,
        pos_embeddings: Optional[Tensor] = None, source_pos_embeddings: Optional[Tensor] = None,
        return_weights=False):
        # self attention
        if self.self_attention:
            x = x + self.droppath_sa(self.ls_sa(self.sa_block(self.norm_sa(add_pos_embeddings(x, pos_embeddings)), mask, attention_mask=sa_mask)))
        # cross attention
        mha_output, att_weights = self.mha_block(
            self.norm_mha(add_pos_embeddings(x, pos_embeddings)), 
            self.norm_encoder(add_pos_embeddings(source_features, source_pos_embeddings)), 
            source_mask=source_mask, return_weights=return_weights)
        x = x + self.droppath_mha(self.ls_mha(mha_output))
        # feed forward
        if self.feedforward:
            x = x + self.droppath_ff(self.ls_ff(self.ff_block(self.norm_ff(x))))
        if return_weights:
            return x, att_weights
        else:
            return x
    

class TransformerEncoderModule(nn.Module):
    def __init__(self, d_model: int, n_layers: int, nhead: int, 
        act=nn.ReLU, attention_dropout=0.0, dropout=0.1, droppath_prob=0.2,
        layer_scale: bool = True, layer_scale_init=0.1,
        shortcut_pos_embeddings: bool = False) -> None:
        super().__init__()
        
        self.shortcut_pos_embeddings = shortcut_pos_embeddings
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model, nhead=nhead,
                act=act,
                attention_dropout=attention_dropout, dropout=dropout, droppath_prob=droppath_prob,
                layer_scale=layer_scale, layer_scale_init=layer_scale_init)
            for _ in range(n_layers)])
   
    def forward(self, features: Tensor, mask: Union[AttentionMask, BoolTensor, None] = None, pos_embeddings: Optional[Tensor] = None):
        mask = AttentionMask.create(mask)
        
        if not self.shortcut_pos_embeddings:
            # only add them at the beginning, not in each attention layer
            features = add_pos_embeddings(features, pos_embeddings)
            pos_embeddings = None

        for encoder_layer in self.encoder_layers:
            features = encoder_layer(
                features, mask=mask, pos_embeddings=pos_embeddings)
        return features


class TransformerTokenDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int,
        n_decoder_layers: int = 0, n_joint_encoder_layers: int = 0, n_output_encoder_layers: int = 0,
        act=nn.ReLU, attention_dropout=0.0, dropout=0.1, droppath_prob=0.2,
        layer_scale: bool = True, layer_scale_init=0.1,
        enc_dec_droppath: bool = False, decoder_sa: bool = True, decoder_ff: bool = True,
        shortcut_tokens: bool = False, shortcut_pos_embeddings: bool = False,
        assign_decode: Optional[str] = None, **assign_kwargs) -> None:
        super().__init__()

        assert assign_decode in (None, 'sha', 'mha')

        self.shortcut_tokens = shortcut_tokens
        self.shortcut_pos_embeddings = shortcut_pos_embeddings

        self.joint_input_encoding = n_joint_encoder_layers > 0
        if self.joint_input_encoding:
            self.joint_encoder_layers = nn.ModuleList([
                TransformerEncoderBlock(
                    d_model=d_model, nhead=nhead,
                    act=act,
                    attention_dropout=attention_dropout, dropout=dropout, droppath_prob=droppath_prob,
                    layer_scale=layer_scale, layer_scale_init=layer_scale_init)
                for _ in range(n_joint_encoder_layers)])

        self.has_decoder = n_decoder_layers > 0
        if self.has_decoder:
            self.decoder_layers = nn.ModuleList([
                TransformerDecoderBlock(
                    d_model=d_model, nhead=nhead,
                    act=act,
                    attention_dropout=attention_dropout, dropout=dropout, droppath_prob=droppath_prob,
                    layer_scale=layer_scale, layer_scale_init=layer_scale_init,
                    encoder_droppath=enc_dec_droppath, self_attention=decoder_sa, feedforward=decoder_ff)
                for _ in range(n_decoder_layers)])
        else:
            assert assign_decode is not None

        self.assign_decoder = get_assign_decoder(assign_decode, d_model=d_model, nhead=nhead, **assign_kwargs)

        self.output_encoding = n_output_encoder_layers > 0
        if self.output_encoding:
            self.output_encoder_layers = nn.ModuleList([
                TransformerEncoderBlock(
                    d_model=d_model, nhead=nhead,
                    act=act,
                    attention_dropout=attention_dropout, dropout=dropout, droppath_prob=droppath_prob,
                    layer_scale=layer_scale, layer_scale_init=layer_scale_init)
                for _ in range(n_output_encoder_layers)])

    def forward(self, 
        token_features: Tensor, region_features: Tensor, 
        token_mask: Union[AttentionMask, BoolTensor, None] = None, 
        region_mask: Union[AttentionMask, BoolTensor, None] = None, 
        token_sa_mask: Union[AttentionMask, BoolTensor, None] = None,
        region_pos_embeddings: Optional[Tensor] = None,
        return_intermediate: bool = False, return_intermediate_attentions: bool = False) \
        -> Tuple[Tensor, Optional[Tensor], Optional[List[Tensor]], Optional[List[Tensor]]]:
        """
        :param token_features: (N x T x d)
        :param region_features: (N x R x d)
        :param token_mask: (N x T) or None
        :param region_mask: (N x R) or None
        :param token_sa_mask: (T_t x T_s) or None
        :param region_pos_embeddings: (N x R x d) or None

        :return token_features, assigment_probs, intermediate_features, cross_attentions:
            - token_features: (N x T x d)
            - assigment_probs: (N x T x R) if assign_decoder is used or return_cross_atts is True, else None
            - intermediate_features: List of (N x T x d), one element for each decoder layer. If no assign_decode and no output encoder is used, then exlusing the last one
            - return_cross_atts: List of (N x T x R), one element for each decoder layer. If no assign_decode is used, then exlusing the last one
        """
        token_mask = AttentionMask.create(token_mask)
        region_mask = AttentionMask.create(region_mask)
        token_sa_mask = AttentionMask.create(token_sa_mask)

        if self.shortcut_tokens:
            # start with the tokens features and add the input token features in each attention layer
            token_embeddings = token_features
            token_features = torch.zeros_like(token_embeddings)
        else:
            # use the processed token features, do not add them in each attention layer
            token_embeddings = None

        if not self.shortcut_pos_embeddings:
            # only add them at the beginning, not in each attention layer
            region_features = add_pos_embeddings(region_features, region_pos_embeddings)
            region_pos_embeddings = None

        # joint input encoder
        if self.joint_input_encoding:
            token_features, region_features = self.joint_encode_input(
                token_features, region_features, 
                token_mask, region_mask,
                token_embeddings, region_pos_embeddings)
        # decoder layers
        if self.assign_decoder is None:
            return_intermediate_attentions = True
        if self.has_decoder:
            token_features, intermediate_features, cross_attentions = self.decode(
                token_features, region_features, 
                token_mask, region_mask, token_sa_mask,
                token_embeddings, region_pos_embeddings,
                return_intermediate=return_intermediate,
                return_cross_atts=return_intermediate_attentions)

        # assignment decoder
        if self.assign_decoder is not None:
            token_features, assigment_probs = self.assign_decoder(
                token_features, region_features, 
                mask=token_mask, source_mask=region_mask,
                pos_embeddings=token_embeddings, source_pos_embeddings=region_pos_embeddings)
        else:
            if return_intermediate_attentions:
                # use the final cross attentions as assignment probs if no assign decoder is used
                *cross_attentions, assigment_probs = cross_attentions
            else:
                assigment_probs = None

        # output encoder
        if self.output_encoding:
            token_features = self.encode_output(token_features, token_mask, token_embeddings)

        if return_intermediate and self.assign_decoder is None and not self.output_encoding:
            # the final intermediate faeture is in this case "token_features", so no need to return it as intermediate
            intermediate_features = intermediate_features[:-1]

        return token_features, assigment_probs, intermediate_features, cross_attentions

    def joint_encode_input(self, 
        token_features, region_features, 
        token_mask, region_mask,
        token_embeddings, region_pos_embeddings):

        N, M_token, d = token_features.shape
        N, M_region, d = region_features.shape

        features = torch.cat([token_features, region_features], dim=1)
        mask = concat_masks(token_mask, region_mask, M_token, M_region, dtype=features.dtype)
        pos_embeddings = concat_pos_embeddings(token_embeddings, region_pos_embeddings, M_token, M_region)

        for joint_encoder_layer in self.joint_encoder_layers:
            features = joint_encoder_layer(
                features, mask=mask, pos_embeddings=pos_embeddings)

        token_features = features[:, :M_token, :]
        region_features = features[:, M_token:, :]
        
        return token_features, region_features

    def decode(self, 
        token_features, region_features, 
        token_mask, region_mask, token_sa_mask,
        token_embeddings, region_pos_embedding,
        return_intermediate: bool = False, return_cross_atts: bool = False):

        intermediate_features = [] if return_intermediate else None
        cross_atts = [] if return_cross_atts else None
        for decoder_layer in self.decoder_layers:
            token_features, cross_att = decoder_layer(
                token_features, region_features, 
                mask=token_mask, source_mask=region_mask, sa_mask=token_sa_mask,
                pos_embeddings=token_embeddings, source_pos_embeddings=region_pos_embedding,
                return_weights=True)
            if return_intermediate:
                intermediate_features.append(token_features)
            if return_cross_atts:
                cross_atts.append(cross_att)

        return token_features, intermediate_features, cross_atts

    def encode_output(self, token_features, token_mask, token_embeddings):
        for encoder_layer in self.output_encoder_layers:
            token_features = encoder_layer(
                token_features, mask=token_mask, pos_embeddings=token_embeddings)
        return token_features


# ========================= Assign Decoders =====================================
def get_assign_decoder(name: Optional[str], d_model, nhead, **kwargs):
    if name is None:
        return None
    elif name == 'mha':
        return MhaAssignDecoder(d_model, nhead=nhead, **kwargs)
    elif name == 'sha':
        return MhaAssignDecoder(d_model, nhead=1, **kwargs)
    else:
        raise ValueError(name)


class MhaAssignDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, skip_assign: bool = False, skip_assign_dropout: float = 0.5):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_source = nn.LayerNorm(d_model)
        self.skip_assign = skip_assign
        if skip_assign:
            self.skip_assign_dropout = nn.Dropout(skip_assign_dropout)

    def forward(self, x, source, 
                mask: Optional[AttentionMask], source_mask: Optional[AttentionMask],
                pos_embeddings, source_pos_embeddings):
        x_in = x
        x = self.norm_x(add_pos_embeddings(x, pos_embeddings))
        source = self.norm_source(add_pos_embeddings(source, source_pos_embeddings))

        x, weights = self.mha(x, source, source, key_padding_mask=AttentionMask.get_inverted_binary_mask(source_mask), need_weights=True)

        if self.skip_assign:
            x = x + self.skip_assign_dropout(x_in)

        if mask is not None:
            x = x * AttentionMask.get_binary_mask(mask).unsqueeze(-1)

        return x, weights