

from math import prod
from typing import Tuple
import einops
from torch import Tensor
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def bbox_l1_ccost(boxes1, boxes2):
    """
    :param boxes1: (... x R_s x 4) in (xc, yc, w, h) format
    :param boxes2: (... x R_t x 4) in (xc, yc, w, h) format
    :return: (... x R_s x R_t)
    """
    assert boxes1.ndim == boxes2.ndim
    assert boxes1.ndim >= 3
    assert boxes1.shape[-1] >= 4 and boxes2.shape[-1] >= 4
    *batch_dims1, R_s, _ = boxes1.shape
    *batch_dims2, R_t, _ = boxes2.shape
    dims = prod(batch_dims1)    
    assert batch_dims1 == batch_dims2
    # (... x R_s x 4)
    boxes1 = boxes1[..., :4].reshape(dims, R_s, 4)
    # (... x R_t x 4)
    boxes2 = boxes2[..., :4].reshape(dims, R_t, 4).to(dtype=boxes1.dtype)
    # (... x R_s x R_t)
    l1_cost = torch.cdist(boxes1, boxes2, p=1)
    return l1_cost.view(*batch_dims1, R_s, R_t)


def bbox_l1_pcost(boxes1, boxes2):
    """
    :param boxes1: (... x 4) in (xc, yc, w, h) format
    :param boxes2: (... x 4) in (xc, yc, w, h) format
    :return: (...)
    """
    boxes1 = boxes1[..., :4]
    boxes2 = boxes2[..., :4]
    assert boxes1.shape == boxes2.shape
    return F.l1_loss(boxes1, boxes2, reduction='none').sum(dim=-1)


def bbox_giou_ccost(boxes1, boxes2):
    """
    :param boxes1: (... x R_s x 4) in (xc, yc, w, h) format
    :param boxes2: (... x R_t x 4) in (xc, yc, w, h) format
    :return: (... x R_s x R_t)
    """
    assert boxes1.ndim == boxes2.ndim
    assert boxes1.ndim >= 3
    assert boxes1.shape[-1] >= 4 and boxes2.shape[-1] >= 4
    *batch_dims1, R_s, _ = boxes1.shape
    *batch_dims2, R_t, _ = boxes2.shape
    assert batch_dims1 == batch_dims2
    dims = prod(batch_dims1)    
    # (... x R_s x 4)
    boxes1 = center_to_corners_format(boxes1[..., :4].reshape(dims, R_s, 4))
    boxes2 = center_to_corners_format(boxes2[..., :4].reshape(dims, R_t, 4).to(dtype=boxes1.dtype))
    # (... x R_s x R_t)
    giou_cost = 1. - box_giou(boxes1, boxes2)
    
    return giou_cost.view(*batch_dims1, R_s, R_t)

def bbox_giou_pcost(boxes1, boxes2):
    """
    :param boxes1: (... x 4) in (xc, yc, w, h) format
    :param boxes2: (... x 4) in (xc, yc, w, h) format
    :return: (...)
    """
    boxes1 = center_to_corners_format(boxes1[..., :4])
    boxes2 = center_to_corners_format(boxes2[..., :4])
    assert boxes1.shape == boxes2.shape
    *batch_dims, _ = boxes1.shape
    boxes1 = boxes1.view(-1, 1, 1, 4)
    boxes2 = boxes2.view(-1, 1, 1, 4)
    # (..., 1, 1)
    giou_cost = 1. - box_giou(boxes1, boxes2)
    return giou_cost.view(*batch_dims)

def box_giou(boxes1, boxes2):
    """
    https://giou.stanford.edu/
    :param boxes1: (... x N x M x 4) in (x0, y0, x1, y1) format
    :param boxes2: (... x N x M x 4) in (x0, y0, x1, y1) format
    :return: (... x N x M) giou
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[..., 2:] >= boxes1[..., :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[..., 2:] >= boxes2[..., :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    
    # (... x N x M) 
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    bottom_right = torch.max(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [...,N,M,2]
    area = width_height[..., :, :, 0] * width_height[..., :, :, 1]
    return iou - (area - union) / area.clamp(min=1e-7)

def box_iou(boxes1, boxes2):
    """
    :param boxes1: (... x N x 4) in (x0, y0, x1, y1) format
    :param boxes2: (... x M x 4) in (x0, y0, x1, y1) format
    :return: (... x N x M) iou and union
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [...,N,M,2]
    right_bottom = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [...,N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [...,N,M,2]
    inter = width_height[..., :, :, 0] * width_height[..., :, :, 1]  # [...,N,M]
    # [...,N,M]
    union = area1[..., :, None] + area2[..., None, :] - inter

    iou = inter / union.clamp(min=1e-7)
    return iou, union


def box_area(boxes: Tensor) -> Tensor:
    """
    :param boxes: (..., 4) in (x0, y0, x1, y1) format
    :return: (...)
    """
    boxes = _upcast(boxes)
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


@torch.no_grad()
def match_multiregions(cost, mask, non_matched_region_mode: str, greedy_match=False):
    """
    :param cost: (N x C x R_s x R_t)
    :param mask: (N x C x R_t)
    """
    N, C, R_s, R_t = cost.shape
    cost = cost.flatten(0, 1) # (N*C x R_s x R_t)
    mask = mask.flatten(0, 1) # (N*C x R_t)
    # List (N*C) of (R_s, R_t') tensors where R_t' is the number of target boxes for the class and sample
    matches_list, assign_mask_list = zip(*[compute_matching(c, m, non_matched_region_mode, greedy_match=greedy_match) for c, m in zip(cost, mask)])

    # (N x C x R_s)
    matches = einops.rearrange(torch.stack(matches_list), '(n c) r_s -> n c r_s', n=N, c=C)
    # (N x C x R_s x R_t)
    assign_mask = einops.rearrange(torch.stack(assign_mask_list), '(n c) r_s r_t -> n c r_s r_t', n=N, c=C)

    return matches, assign_mask


def compute_matching(cost: torch.FloatTensor, mask: torch.BoolTensor, non_matched_region_mode: str, greedy_match=False):
    device = cost.device
    R_t_all = mask.shape[-1]
    cost = cost[:, mask]
    if not greedy_match:
        cost = cost.cpu()

    R_s, R_t = cost.shape
    if R_t == 0:
        return torch.zeros((R_s,), dtype=torch.bool, device=device), torch.zeros((R_s, R_t_all), dtype=torch.bool, device=device)

    if non_matched_region_mode == 'balanced_match':
        cost = extend_cost_for_balanced_match(cost)
    else:
        assert non_matched_region_mode == 'ignore', f'Unknown non_matched_region_mode {non_matched_region_mode}'

    if greedy_match:
        indices_s, indices_t = _do_matching_greedy(cost)
    else:
        indices_s, indices_t = _do_matching(cost)
    # (N_match)
    matches = indices_s[indices_t < R_t]
    # (N_match)
    indices_t = indices_t % R_t  # for balanced match

    # (R_s)
    matches: torch.BoolTensor = torch.zeros((R_s,), dtype=torch.bool, device=device if greedy_match else 'cpu').scatter_(0, matches, True)
    # (R_s x R_t)
    # based on both, indices_s and indices_t
    assign_mask = torch.zeros((R_s, R_t), dtype=torch.bool, device=device if greedy_match else 'cpu')
    assign_mask[indices_s, indices_t] = True

    if not greedy_match:
        matches = matches.to(device, non_blocking=True)
        assign_mask = assign_mask.to(device, non_blocking=True)

    assign_mask_all = torch.zeros((R_s, R_t_all), dtype=torch.bool, device=device)
    assign_mask_all[:, mask] = assign_mask

    return matches, assign_mask_all


def _do_matching(cost: torch.FloatTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    indices_s, indices_t = linear_sum_assignment(cost.numpy())
    indices_s = torch.from_numpy(indices_s)
    indices_t = torch.from_numpy(indices_t)
    return indices_s, indices_t


def _do_matching_greedy(cost: torch.FloatTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    :param R_s, R_t
    :return: (N_match), (N_match)
    """
    indices_s, indices_t = [], []
    R_s, R_t = cost.shape
    cost = cost.clone()

    for _ in range(min(R_s, R_t)):
        mins_R_t, argmins_R_t = cost.min(dim=1)
        s = mins_R_t.argmin()
        t = argmins_R_t[s]
        indices_s.append(s)
        indices_t.append(t)
        cost[s, :] = float('inf')
        cost[:, t] = float('inf')

    indices_s = torch.stack(indices_s)
    indices_t = torch.stack(indices_t)
    return indices_s, indices_t


def extend_cost_for_balanced_match(cost):
    # cost: (R_s x R_t')
    R_s, R_t = cost.shape
    max_cost_diff = cost.max() - cost.min()

    if R_s > R_t:
        n_copies = R_s // R_t + (1 if R_s % R_t != 0 else 0)
        R_t_total = R_t * n_copies
        assert R_t_total >= R_s

        # (n_copies - 1)
        extra_costs = torch.full((n_copies - 1,), max_cost_diff + 1., 
                                 dtype=torch.float, device=cost.device).cumsum(0)
        # (R_t_total)
        extra_costs = torch.cat([
            torch.zeros((R_t,), dtype=torch.float, device=cost.device),
            extra_costs.repeat_interleave(R_t)
        ])
        # (R_s x R_t_total)
        cost = cost.repeat(1, n_copies) + extra_costs[None, :]

    return cost


# ----------------- Conversion utils -----------------
def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h, *args = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h), *args]
    return torch.stack(b, dim=-1)

def corners_to_center_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of corners format (x_0, y_0, x_1, y_1) to center format
    (center_x, center_y, width, height).
    """
    x_0, y_0, x_1, y_1, *args = x.unbind(-1)
    b = [(x_0 + x_1) / 2, (y_0 + y_1) / 2, (x_1 - x_0), (y_1 - y_0), *args]
    return torch.stack(b, dim=-1)

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
