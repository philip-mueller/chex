from typing import Optional
import torch


def subsample_classes(cls_mask: Optional[torch.BoolTensor], expected_subssample_count: Optional[int], is_active: bool, C: int, device) -> torch.BoolTensor:
    cls_mask = cls_mask.any(0) if cls_mask is not None else torch.ones((C,), dtype=torch.bool, device=device)
    if expected_subssample_count is None:
        return cls_mask
    if not is_active:
        return torch.zeros_like(cls_mask)

    num_classes = cls_mask.sum().float().clamp_min(1)
    selection_prob = (expected_subssample_count / num_classes.float()).clamp(0, 1)
    selection_mask = torch.rand_like(cls_mask, dtype=torch.float) < selection_prob

    # (C)
    default_selection_mask = cls_mask & (cls_mask.cumsum(0) <= expected_subssample_count)
    selection_mask = cls_mask & selection_mask
    # (1)
    empty_selection = selection_mask.sum() == 0
    return torch.where(empty_selection, default_selection_mask, selection_mask)

def subsample_anat_regions(anat_mask: Optional[torch.BoolTensor], expected_subssample_count: Optional[int], is_active: bool, A: int, device) -> torch.BoolTensor:
    assert anat_mask is None or anat_mask.ndim == 2
    return subsample_classes(anat_mask, expected_subssample_count, is_active, A, device)

def subsample_anat_classes(target_anat_cls_labels: Optional[torch.BoolTensor], expected_anat_subssample_count: Optional[int], expected_cls_subssample_count: Optional[int], is_active: bool, C: int, A: int, device) -> torch.BoolTensor:
    if target_anat_cls_labels is None:
        return torch.ones((A, C), dtype=torch.bool, device=device)
    assert target_anat_cls_labels.ndim == 3
    # (C)
    cls_mask = subsample_classes(target_anat_cls_labels.any(1), expected_cls_subssample_count, is_active, C, device)
    # (A)
    anat_mask = subsample_anat_regions(target_anat_cls_labels.any(2), expected_anat_subssample_count, is_active, A, device)
    # (A x C)
    return anat_mask[:, None] & cls_mask[None, :]

def subsample_anat_neg_classes_balanced(targets: torch.BoolTensor, has_targets: torch.BoolTensor, expected_subssample_count: int, sample_only_positives: bool = False) -> torch.BoolTensor:
    """
    targets: (N x R x C)
    has_targets: (N x R x C)
    """
    pos_count = targets.sum(dim=(0, 1))
    # (C)
    cls_mask = has_targets.any(1).any(0)
    C = cls_mask.shape[0]
    if C <= expected_subssample_count:
        return cls_mask
    
    # (C)
    # make sure that 
    weights = pos_count.float()
    EPS = 1e-7
    # EPS makes sure that we always have enough classes to sample from (wo replacement) but the non-positive classes are still sampled with almost 0 prob
    # (expected_subssample_count)
    sampled_classes: torch.LongTensor = torch.multinomial(weights + EPS , expected_subssample_count, replacement=False)

    selection_mask = torch.zeros_like(cls_mask, dtype=torch.bool) # .scatter(0, sampled_classes, True)
    selection_mask[sampled_classes] = True
    selection_mask = selection_mask & cls_mask
    if sample_only_positives:
        selection_mask = selection_mask & (pos_count > 0)
    return selection_mask
