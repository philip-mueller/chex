from functools import partial
from math import prod
from typing import Optional, Tuple
import einops
import torch
import torch.nn.functional as F

def classify_features(features: torch.FloatTensor, pos_prompt_emb: torch.FloatTensor, neg_prompt_emb: torch.FloatTensor, 
            normalized: bool = True, temp=1.0, softmax=False, 
            threshold: Optional[float] = 0.5, return_logits=False) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """
        :param features: (N x ... x d)
        :param pos_prompt_emb: (N x ... x d)
        :param neg_prompt_emb: (N x ... x d)
        :return (N x ...)
        """
        assert pos_prompt_emb.ndim == neg_prompt_emb.ndim
        assert features.ndim == pos_prompt_emb.ndim, f'{features.ndim} != {pos_prompt_emb.ndim}'
        features, pos_prompt_emb, neg_prompt_emb = torch.broadcast_tensors(features, pos_prompt_emb, neg_prompt_emb)

        if normalized:
            features = F.normalize(features, dim=-1)
            pos_prompt_emb = F.normalize(pos_prompt_emb, dim=-1)
            neg_prompt_emb = F.normalize(neg_prompt_emb, dim=-1)
        else:
            features = features.contiguous()
            pos_prompt_emb = pos_prompt_emb.contiguous()
            neg_prompt_emb = neg_prompt_emb.contiguous()

       
        N, *dims, d = features.shape
        n_dims = prod(dims)
        features = features.view(N, n_dims, d)
        pos_prompt_emb = pos_prompt_emb.view(N, n_dims, d)
        neg_prompt_emb = neg_prompt_emb.view(N, n_dims, d)

        pos_logits = torch.einsum('ijd,ijd->ij', features, pos_prompt_emb) / temp # (N x dims)
        neg_logits = torch.einsum('ijd,ijd->ij', features, neg_prompt_emb) / temp # (N x dims)

        if softmax:
            # (N x dims x 2)
            probs = torch.softmax(torch.stack([pos_logits, neg_logits], dim=-1), dim=-1)
            probs = probs[..., 0]  # only positive probs 
        else:
            probs = torch.sigmoid(pos_logits - neg_logits)
        probs = probs.view(N, *dims)

        preds = probs > threshold if threshold is not None else torch.ones_like(probs, dtype=bool)

        if not return_logits:
            return probs, preds

        if softmax:
            logits = torch.log_softmax(torch.stack([pos_logits, neg_logits], dim=-1), dim=-1)
            logits = logits[..., 0]  # only positive probs 
        else:
            logits = pos_logits - neg_logits
        logits = logits.view(N, *dims)

        return logits, probs, preds



# ----------------------- Binary (and multilabel binary) losses ----------------------- #
def binary_focal_loss_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2) -> torch.Tensor:
    logits, targets = torch.broadcast_tensors(logits, targets)
    targets = targets.to(dtype=logits.dtype)
    
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if torch.is_tensor(alpha) or alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss

def binary_focal_loss_probs(
        probs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        eps=1e-7) -> torch.Tensor:
    probs, targets = torch.broadcast_tensors(probs, targets)
    targets = targets.to(dtype=probs.dtype)
    
    p = probs
    with torch.autocast(device_type='cuda', enabled=False):
        ce_loss = F.binary_cross_entropy(probs.float().clamp(min=eps, max=1-eps), targets.float(), reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def autocompute_class_weights(targets: torch.Tensor, per_class: bool = False, cls_dim: int = None) -> torch.Tensor:
    if per_class:
        assert cls_dim is not None
        dims_before = targets.shape[:cls_dim]
        C = targets.shape[cls_dim]
        dims_after = targets.shape[cls_dim+1:]
        # (... x C x ...) where C is the dim at index cls_dim
        targets = targets.view(prod(dims_before), C, prod(dims_after))

        N_pos = targets.sum(dim=0).sum(dim=-1)  # (C)
        # (1 x ... x 1 x C x 1 x ...)
        N_pos = N_pos.view((1,) * len(dims_before) + (C,) + (1,) * len(dims_after))
        N = targets.numel() / C  # ()
    else:
        # (...)
        targets = targets.view(-1)
        N_pos = targets.sum() 
        N = targets.numel() 

    N_neg = N - N_pos  # (C) or ()

    weight_pos = (N + 1) / (N_pos + 1)  # (C) or ()
    weight_neg = (N + 1) / (N_neg + 1)  # (C) or ()

    return weight_pos, weight_neg


def get_focal_loss(logits: bool = True, auto_weight: bool = False, **kwargs):
    focal_loss_fn = binary_focal_loss_logits if logits else binary_focal_loss_probs
    if auto_weight:
        def _loss_fn(preds, targets):
            preds, targets = torch.broadcast_tensors(preds, targets)
            weight_pos, weight_neg = autocompute_class_weights(targets, per_class=False)
            alpha = weight_pos / (weight_pos + weight_neg)
            return focal_loss_fn(preds, targets, alpha=alpha, **kwargs)
    else:
        _loss_fn = partial(focal_loss_fn, **kwargs)
    return _loss_fn
