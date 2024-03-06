from typing import Collection, Optional
from model.components.transformer import AttentionMask
import torch
import torch.nn.functional as F
import einops

from model.supervisors.utils import subsample_anat_neg_classes_balanced


def global_contrastive_loss(z1, z2, temp: float):
    """
    :param z1: (N x d)
    :param z2: (N x d)
    :param temp: temperature
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    similarities = torch.einsum('nd,md->nm', z1, z2) / temp
    positives = torch.diagonal(similarities)

    loss_12 = - positives + torch.logsumexp(similarities, dim=1)
    loss_21 = - positives + torch.logsumexp(similarities, dim=0)

    return loss_12.mean() + loss_21.mean()


def sentence_contrastive_loss(z_sent, z_reg, sentence_mask,
                              temp: float, 
                              cross_sample_negatives: bool, 
                              lambda_sent2reg: float, lambda_reg2sent: float):
    """
    :param z_sent: (N x S x d)
    :param z_reg: (N x S x d)
    :param mask: (N x S)
    :param temp: temperature
    :param cross_sample_negatives: whether to use cross sample negatives
    :param lambda_sent2reg: weight for sent2reg loss
    :param lambda_reg2sent: weight for reg2sent loss
    """
    # (N x S x d)
    z_sent = F.normalize(z_sent, dim=-1)
    z_reg = F.normalize(z_reg, dim=-1)

    if cross_sample_negatives:
        N, S = z_sent.shape[:2]
        # (N x S x N' x S')
        sim = torch.einsum('isd,jrd->isjr', z_sent, z_reg) / temp
        # (N x S)
        pos = sim.view(N*S, N*S).diagonal().view(N, S)
        # (N x S x N' x S')
        neg_sent_reg = sim + AttentionMask.get_additive_mask(sentence_mask, dtype=sim.dtype)[None, None, :, :]
        # (N x S x N'*S')
        neg_sent_reg = neg_sent_reg.view(N, S, N*S)
        # (N x S x N' x S')
        neg_reg_sent = einops.rearrange(sim, 'i s j r -> j r i s') + AttentionMask.get_additive_mask(sentence_mask, dtype=sim.dtype)[None, None, :, :]
        neg_reg_sent = neg_reg_sent.view(N, S, N*S)
    else:
        # (N x S x S')
        sim = torch.einsum('nsd,nrd->nsr', z_sent, z_reg) / temp
        # (N x S)
        pos = torch.diagonal(sim, dim1=-2, dim2=-1)
        # (N x S x S')
        neg_sent_reg = sim + AttentionMask.get_additive_mask(sentence_mask, dtype=sim.dtype)[:, None, :]
        # (N x S x S')
        neg_reg_sent = sim.transpose(1, 2) + AttentionMask.get_additive_mask(sentence_mask, dtype=sim.dtype)[:, None, :]

    # (N x S)
    loss_sent_reg = - pos + torch.logsumexp(neg_sent_reg, dim=-1)
    loss_reg_sent = - pos + torch.logsumexp(neg_reg_sent, dim=-1)

    sentence_weights = sentence_mask.to(dtype=z_sent.dtype)
    sentence_weights = sentence_weights / sentence_weights.sum(dim=-1, keepdim=True).clamp(min=1e-7)
    loss_sent_reg = (loss_sent_reg * sentence_weights).sum(dim=-1)
    loss_reg_sent = (loss_reg_sent * sentence_weights).sum(dim=-1)
    return lambda_sent2reg * loss_sent_reg.mean() + lambda_reg2sent * loss_reg_sent.mean()


def sentence_mse_loss(z_sent, z_reg, sentence_mask):
    """
    :param z_sent: (N x S x d)
    :param z_reg: (N x S x d)
    :param mask: (N x S)
    """
    # (N x S)
    mse_loss = F.mse_loss(z_sent, z_reg, reduction='none').sum(dim=-1)
    sentence_weights = sentence_mask.to(dtype=z_sent.dtype)
    sentence_weights = sentence_weights / sentence_weights.sum(dim=-1, keepdim=True).clamp(min=1e-7)
    return (mse_loss * sentence_weights).sum(dim=-1).mean()



def pathology_contrastive_loss(features: torch.FloatTensor, pos_prompt_emb: torch.FloatTensor, neg_prompt_emb: torch.FloatTensor, targets: torch.BoolTensor,
                               temp, normalized: bool = True, use_negatives_of_other_classes: bool = True, ignore_other_classes_when_false_target: bool = False):
    """
    :param features: (N x C x d)
    :param pos_prompt_emb: (C x d)
    :param neg_prompt_emb: (C x d)
    :param targets: (N x C)
    """
    assert features.ndim == 3 and pos_prompt_emb.ndim == 2 and neg_prompt_emb.ndim == 2 and targets.ndim == 2

    N, C = targets.shape
    if normalized:
        pos_prompt_emb = F.normalize(pos_prompt_emb, dim=-1)
        neg_prompt_emb = F.normalize(neg_prompt_emb, dim=-1)
        features = F.normalize(features, dim=-1)

    # (N x C x C)
    pos_scores = torch.einsum('ncd,ed->nce', features, pos_prompt_emb) / temp
    neg_scores = torch.einsum('ncd,ed->nce', features, neg_prompt_emb) / temp

    # (1 x C)
    pos_index_for_positives = torch.arange(C, device=features.device)[None, :]
    pos_index_for_negatives = torch.arange(C, 2*C, device=features.device)[None, :]
    # (N x C)
    pos_index = targets * pos_index_for_positives + (1 - targets) * pos_index_for_negatives

    # (N x C x C)
    other_classes_mask = torch.ones_like(neg_scores, dtype=torch.bool)
    other_classes_mask.diagonal(dim1=-2, dim2=-1)[:, :] = False

    if not use_negatives_of_other_classes:
        neg_scores = neg_scores.masked_fill(other_classes_mask, -torch.inf)
    if ignore_other_classes_when_false_target:
        other_classes_when_false = other_classes_mask & ~(targets[..., None].bool())
        pos_scores = pos_scores.masked_fill(other_classes_when_false, -torch.inf)
        neg_scores = neg_scores.masked_fill(other_classes_when_false, -torch.inf)

    # (N x C x 2C)
    all_scores = torch.cat([pos_scores, neg_scores], dim=-1)
    all_scores = einops.rearrange(all_scores, 'n c c2 -> (n c) c2')
    pos_index = einops.rearrange(pos_index, 'n c -> (n c)')
    # (N*C)
    loss = F.cross_entropy(all_scores, pos_index, reduction='none')
    return loss.view(N, C)


def anat_pathology_contrastive_loss(features: torch.FloatTensor, 
                                        pos_prompt_emb: torch.FloatTensor, 
                                        neg_prompt_emb: torch.FloatTensor, 
                                        region_pos_prompt_emb: Optional[torch.FloatTensor],
                                        region_neg_prompt_emb: Optional[torch.FloatTensor],
                                        targets: torch.BoolTensor,
                                        has_targets: torch.BoolTensor,
                                   temp, normalized: bool = True, 
                                   positive_prompt_examples: Collection[str] = ('pos', 'region_pos', 'neg', 'region_neg'),
                                   negative_prompt_examples: Collection[str] = ('pos', 'region_pos', 'other_region_pos', 'neg', 'region_neg', 'other_region_neg'),
                                   subsample_negatives: Optional[int] = None,
                                   ):
    """
    :param features: (N x R x d)
    :param pos_prompt_emb: (C x d)
    :param neg_prompt_emb: (C x d)
    :param region_pos_prompt_emb: (R x C x d)
    :param region_neg_prompt_emb: (R x C x d)
    :param targets: (N x R x C)
    :param has_targets: (N x R x C)
    :param positive_prompt_examples: which prompts to use as positive contrastive examples  (i.e. with a pulling force)
        - pos: pull to pos_prompt_emb, only if the class is present in targets
            (e.g. "pneumonia" if pneumonia is present in the region)
        - neg: pull to neg_prompt_emb, only if the class is not present in targets
            (e.g. "no pneumonia" if pneumonia is not present in the region)
        - region_pos: pull to region_pos_prompt_emb, only if the class is present in targets and use the region itself 
            (e.g. "pneumonia in the left lung" if the region is the left lung and pneumonia is present in the region)
        - region_neg: pull to region_neg_prompt_emb, only if the class is not present in targets and only for the region itself
            (e.g. "no pneumonia in the left lung" if the region is the left lung and pneumonia is not present in the region)
    :param negative_prompt_examples: which prompts to use as negative contrastive examples (i.e. with a pushing force)
        - pos: push away from pos_prompt_emb, only if the class is not present in targets
            (e.g. "pneumonia" if pneumonia is not present in the region)
        - neg: push away from neg_prompt_emb, only if the class is present in targets
            (e.g. "no pneumonia" if pneumonia is present in the region)
        - region_pos: push away from region_pos_prompt_emb, only if the class is not present in targets and use the region itself
            (e.g. "pneumonia in the left lung" if the region is the left lung and pneumonia is not present in the region)
        - region_neg: push away from region_neg_prompt_emb, only if the class is present in targets and only for the region itself
            (e.g. "no pneumonia in the left lung" if the region is the left lung and pneumonia is present in the region)
        - other_region_pos: push away from region_pos_prompt_emb, for all other regions
            (e.g. "pneumonia in the right lung" if the region is the left lung, pneumonia may or may not be present in either lung)
        - other_region_neg: push away from region_neg_prompt_emb, for all other regions
            (e.g. "no pneumonia in the right lung" if the region is the left lung, pneumonia may or may not be present in either lung)
    """
    assert all(p in ('pos', 'neg', 'region_pos', 'region_neg') for p in positive_prompt_examples)
    assert all(p in ('pos', 'neg', 'region_pos', 'region_neg', 'other_region_pos', 'other_region_neg') for p in negative_prompt_examples)
    assert features.ndim == 3
    N, R, d = features.shape
    assert targets.ndim == 3 and targets.shape[:2] == (N, R)
    C = targets.shape[-1]
    assert pos_prompt_emb.shape == neg_prompt_emb.shape == (C, d)
    targets = targets.bool()
    has_targets = has_targets.bool()
    has_region = has_targets.any(dim=-1)

    use_region_prompts = any(p in positive_prompt_examples or p in negative_prompt_examples
                            for p in ('region_pos', 'region_neg', 'other_region_pos', 'other_region_neg'))
    
    # list of tensors (N x R x K) where K is the number of prompts (pos or neg)
    scores = []
    pos_mask = []
    neg_mask = []

    if normalized:
        pos_prompt_emb = F.normalize(pos_prompt_emb, dim=-1)
        neg_prompt_emb = F.normalize(neg_prompt_emb, dim=-1)
        features = F.normalize(features, dim=-1)

    if use_region_prompts:
        assert region_pos_prompt_emb is not None and region_neg_prompt_emb is not None
        assert region_pos_prompt_emb.shape == region_neg_prompt_emb.shape == (R, C, d), f'{region_pos_prompt_emb.shape} != {region_neg_prompt_emb.shape} != {(R, C, d)}'
        if normalized:
            region_pos_prompt_emb = F.normalize(region_pos_prompt_emb, dim=-1)
            region_neg_prompt_emb = F.normalize(region_neg_prompt_emb, dim=-1)

    if subsample_negatives is not None:
        # (C)
        neg_base_mask = subsample_anat_neg_classes_balanced(targets, has_targets, subsample_negatives, sample_only_positives=False)
        # (1 x 1 x C)
        neg_base_mask = neg_base_mask[None, None, :]
    else:
        neg_base_mask = torch.ones_like(targets)


    # --> Positive prompts
    if 'pos' in positive_prompt_examples or 'pos' in negative_prompt_examples:
        # (N x R x C)
        scores.append(torch.einsum('nrd,cd->nrc', features, pos_prompt_emb))
        pos_mask.append(targets & has_targets if 'pos' in positive_prompt_examples else torch.zeros_like(targets))
        neg_mask.append(~targets & has_targets & neg_base_mask if 'pos' in negative_prompt_examples else torch.zeros_like(targets))
    if 'neg' in positive_prompt_examples or 'neg' in negative_prompt_examples:
        # (N x R x C)
        scores.append(torch.einsum('nrd,cd->nrc', features, neg_prompt_emb))
        pos_mask.append(~targets & has_targets & neg_base_mask if 'neg' in positive_prompt_examples else torch.zeros_like(targets))
        neg_mask.append(targets & has_targets if 'neg' in negative_prompt_examples else torch.zeros_like(targets))
    if 'region_pos' in positive_prompt_examples or 'region_pos' in negative_prompt_examples:
        # (N x R x C)
        scores.append(torch.einsum('nrd,rcd->nrc', features, region_pos_prompt_emb))
        pos_mask.append(targets & has_targets if 'region_pos' in positive_prompt_examples else torch.zeros_like(targets))
        neg_mask.append(~targets & has_targets & neg_base_mask if 'region_pos' in negative_prompt_examples else torch.zeros_like(targets))
    if 'region_neg' in positive_prompt_examples or 'region_neg' in negative_prompt_examples:
        # (N x R x C)
        scores.append(torch.einsum('nrd,rcd->nrc', features, region_neg_prompt_emb))
        pos_mask.append(~targets & has_targets & neg_base_mask if 'region_neg' in positive_prompt_examples else torch.zeros_like(targets))
        neg_mask.append(targets & has_targets if 'region_neg' in negative_prompt_examples else torch.zeros_like(targets))
    if 'other_region_pos' in negative_prompt_examples:
        # (N x R x R' x C)
        other_reg_scores = torch.einsum('nrd,kcd->nrkc', features, region_pos_prompt_emb)
        other_reg_scores = einops.rearrange(other_reg_scores, 'n r k c -> n r (k c)')
        scores.append(other_reg_scores)
        pos_mask.append(targets.new_zeros((N, R, R * C)))
        # (R x R')
        other_reg_mask = ~torch.eye(R, device=targets.device, dtype=torch.bool)
        other_reg_mask = einops.repeat(other_reg_mask, 'r1 r2 -> n r1 (r2 c)', n=N, c=C)
        has_other_reg_targets = has_targets[:, :, None, :] & has_targets[:, None, :, :]
        has_other_reg_targets = has_other_reg_targets.flatten(2, 3)
        neg_mask.append(other_reg_mask & has_other_reg_targets)
    if 'other_region_neg' in negative_prompt_examples:
        # (N x R x R' x C)
        other_reg_scores = torch.einsum('nrd,kcd->nrkc', features, region_neg_prompt_emb)
        other_reg_scores = einops.rearrange(other_reg_scores, 'n r k c -> n r (k c)')
        scores.append(other_reg_scores)
        pos_mask.append(targets.new_zeros((N, R, R * C)))
        # (R x R')
        other_reg_mask = ~torch.eye(R, device=targets.device, dtype=torch.bool)
        other_reg_mask = einops.repeat(other_reg_mask, 'r1 r2 -> n r1 (r2 c)', n=N, c=C)
        has_other_reg_targets = has_targets[:, :, None, :] & has_targets[:, None, :, :]
        has_other_reg_targets = has_other_reg_targets.flatten(2, 3)
        neg_mask.append(other_reg_mask & has_other_reg_targets)

    # (N x R x K)
    scores = torch.cat(scores, dim=-1) / temp
    pos_mask = torch.cat(pos_mask, dim=-1)
    neg_mask = torch.cat(neg_mask, dim=-1)
    # in contrastive learning we also use the positive examples as contrasts to imply a uniform distribution
    neg_mask = neg_mask | pos_mask
    # (N x R)
    pos_loss = - (pos_mask * scores).sum(dim=-1) / pos_mask.sum(dim=-1).clamp_min(1)
    # print('neg mask', neg_mask.sum(1).min(), neg_mask.sum(1).max(), neg_mask.sum(1).float().mean())
    # print('pos mask', pos_mask.sum(1).min(), pos_mask.sum(1).max(), pos_mask.sum(1).float().mean())
    neg_scores = scores.masked_fill(~neg_mask, float('-inf')).masked_fill(~has_region[:, :, None], 0)
    neg_loss = torch.logsumexp(neg_scores, dim=-1)
    # (N)
    return has_region * (pos_loss + neg_loss)
