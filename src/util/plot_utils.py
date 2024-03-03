
from collections import defaultdict
from itertools import islice, repeat
import os
from typing import List, Optional, Union
import einops
from matplotlib import patches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from torch import Tensor
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap


def prepare_wandb_bbox_images(
    images: Tensor,
    preds: Union[Tensor, List[Tensor]],
    targets: Union[Tensor, List[Tensor]],
    class_names: List[str],
    predicted_box_masks: Optional[Tensor] = None,
    predicted_box_mask_logits: Optional[Tensor] = None,
    target_box_masks: Optional[Tensor] = None,
    one_box_per_class: bool = True,
    max_samples: int = 10
):
    """
    :param images: List of images to log (N x C x H x W)
    :param preds: List of Tensors with predicted bounding boxes
                  (N x K x [x_c, y_c, w, h, class_id, confidence])
            or 
            (N x K x x_)
    :param targets: List of Tensors with target bounding boxes
                    (N x M x x, y, w, h, class_id)
    :param seg_mask_from_rois: segmentation masks derived from ROIs.
        If not None, will be plotted as "predictions".
        (N x H x W) with integer values corresponding to (0-based) indices of
        class_names. For no-finding/bg use the index len(class_names)-1.
    :param seg_mask_from_patches: segmentation masks derived from patches.
        If not None, will be plotted as "patch_preds".
        (N x H x W) with integer values corresponding to (0-based) indices of
        class_names. For no-finding/bg use the index len(class_names)-1.
    :param class_names: Names of the classes (including the no-finding/bg
                        class name as the last class)
    """
    assert len(images) == len(preds)
    assert targets is None or len(images) == len(targets)
    preds, targets = prepare_boxes_for_plotting(preds, targets, class_names, predicted_box_masks, predicted_box_mask_logits, target_box_masks, one_box_per_class)
    return [
        wandb.Image(
            image,
            boxes=wandb_prepare_bboxes(pred, target, class_names),
        ) for image, pred, target in islice(zip(images, preds, targets), max_samples)
    ]

def prepare_boxes_for_plotting(preds, targets, class_names, predicted_box_masks, predicted_box_mask_logits, target_box_masks, one_box_per_class):
    if one_box_per_class:
        preds = preds.float()
        targets = targets.float() if targets is not None else None
        N, K, dim_box = preds.shape
        assert dim_box == 4
        assert K == len(class_names)
        assert targets is None or preds.shape == targets.shape
        assert targets is None or target_box_masks is not None
        if predicted_box_masks is None:
            predicted_box_masks = torch.ones((N, K), dtype=torch.bool, device=preds.device)
            predicted_box_mask_logits = torch.zeros((N, K), dtype=torch.float, device=preds.device)
        predicted_box_masks = predicted_box_masks.bool()
        predicted_box_mask_logits = predicted_box_mask_logits.float()
        target_box_masks = target_box_masks.bool() if targets is not None else None

        # (N x K)
        class_ids = einops.repeat(torch.arange(len(class_names), dtype=torch.float, device=preds.device), 'k -> n k', n=N)
        # (N x K)
        confidence = predicted_box_mask_logits.sigmoid()
        # (N x K x 6)
        preds = torch.cat([preds, class_ids[:, :, None], confidence[:, :, None]], dim=2).numpy()
        # (N x K x 5)
        targets = torch.cat([targets, class_ids[:, :, None]], dim=2).numpy() if targets is not None else None

        preds = [pred[mask, :] if mask.sum() > 0 else np.zeros((0, 6)) for pred, mask in zip(preds, predicted_box_masks)]
        targets = [trgt[mask, :] for trgt, mask in  zip(targets, target_box_masks)] if targets is not None else repeat(None)
    else:
        assert all(pred.shape[-1] == 6 for pred in preds)
        assert targets is None or all(target.shape[-1] == 5 for target in targets)
        preds = [pred.float().numpy() for pred in preds]
        targets = [target.float().numpy() for target in targets] if targets is not None else repeat(None)
    return preds,targets

def wandb_prepare_bboxes(preds: np.ndarray, targets: np.ndarray, class_names: List[str]):
    """
    Convert bboxes to wandb format

    :param preds: Tensor with predicted bounding boxes
                  (K x x_c, y_c, w, h, class_id, confidence)
    :param targets: Tensor with target bounding boxes
                    (K x x_c, y_c, w, h, class_id)
    :param class_names: List to map class_id to class_name
    """
    bboxes = defaultdict(lambda: defaultdict(list))

    # Add predicted bboxes
    for pred in preds:
        x_c, y_c, w, h, class_id, confidence = [p.item() for p in pred]
        class_id = int(class_id)
        bboxes['predictions']['box_data'].append({
            'position': {
                'minX': x_c - w / 2.,
                'minY': y_c - h / 2.,
                'maxX': x_c + w / 2.,
                'maxY': y_c + h / 2.,
            },
            'class_id': class_id,
            'box_caption': class_names[class_id],
            'scores': {'confidence in %': confidence * 100}
        })
        bboxes['predictions']['class_labels'] = {
            i: name for i, name in enumerate(class_names)
        }

    # Add target bboxes
    if targets is not None:
        for target in targets:
            x_c, y_c, w, h, class_id = [t.item() for t in target]
            class_id = int(class_id)
            bboxes['targets']['box_data'].append({
                'position': {
                    'minX': x_c - w / 2.,
                    'minY': y_c - h / 2.,
                    'maxX': x_c + w / 2.,
                    'maxY': y_c + h / 2.,
                },
                'class_id': class_id,
                'box_caption': class_names[class_id]
            })
            bboxes['targets']['class_labels'] = {
                i: name for i, name in enumerate(class_names)
            }

    return dict(bboxes)
    

COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#e377c2', '#ff7f0e', '#17becf', '#8c564b']

def color_map_for_classes(class_names, cmap=None):
    if len(class_names) <= len(COLORS):
        color_list = COLORS[:len(class_names)]
    else:
        cmap = plt.cm.get_cmap(cmap)
        color_list = cmap(np.linspace(0, 1, len(class_names)))
    return color_list


def plot_img_with_bounding_boxes(ax, class_names: list, class_cmap, img,
                                 pred_list=None, target_list=None,
                                 show_classes=False, plot_gt=True, plot_pred=True,
                                 threshold: float = 0.0, annotate_boxes=True):
    if torch.is_tensor(img):
        # Convert to numpy and denormalize
        img = img.cpu().numpy()
        img = img * 4.8828125e-4
        img = img + 0.5
        np.clip(img, 0, 1, out=img)
    img_size = img.shape[:2]
    ax.imshow(img, cmap='gray')
    if pred_list is not None and plot_pred:
        if torch.is_tensor(pred_list):
            pred_list = pred_list.detach().cpu().numpy()
        for box_prediction in pred_list:
            draw_box(ax, box_prediction, class_names, class_cmap, is_gt=False, img_size=img_size, threshold=threshold)
    if target_list is not None and plot_gt:
        if torch.is_tensor(target_list):
            target_list = target_list.detach().cpu().numpy()
        for box_prediction in target_list:
            draw_box(ax, box_prediction, class_names, class_cmap, is_gt=True, img_size=img_size, annotate_box=annotate_boxes)
    handles = [Line2D([0], [0], label='Target', color='black', linestyle = 'dashed', alpha=0.8),
               Line2D([0], [0], label='Pred', color='black', alpha=0.8)]
    if show_classes:
        handles += [patches.Patch(
            color=to_rgba(class_cmap[i], alpha=0.3),
            label=name
        ) for i, name in enumerate(class_names)]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def draw_box(ax, box_prediction, class_names: list, class_cmap, is_gt: bool, img_size, threshold: float = 0.0, annotate_box=True):
    if torch.is_tensor(box_prediction):
        box_prediction = box_prediction.detach().cpu().numpy()
    if is_gt:
        x_c, y_c, w, h, class_id = box_prediction
    else:
        x_c, y_c, w, h, class_id, confidence = box_prediction
        if confidence < threshold:
            return
    x = x_c - w / 2
    y = y_c - h / 2
    x *= img_size[1]
    y *= img_size[0]
    w *= img_size[1]
    h *= img_size[0]
    class_name = class_names[int(class_id.item())]
    color = class_cmap[int(class_id.item())]

    ax.add_patch(patches.Rectangle(
        (x, y), w, h,
        fill=True,
        facecolor=to_rgba(color, alpha=0.1),
        edgecolor=to_rgba(color, alpha=1.0),
        linestyle = 'dashed' if is_gt else 'solid',
        lw=1
    ))
    if annotate_box:
        if is_gt:
            ax.annotate(class_name, xy=(x, y), xytext=(2, 2),
                        textcoords='offset points', ha='left', va='bottom',
                        fontsize=10, color='white',
                        bbox={"facecolor": to_rgba(color, alpha=1.0),
                            "alpha": 1.0,
                            'pad': 2,
                            'edgecolor': 'none'})
        else:
            ax.annotate(f'{class_name}: {confidence:.2f}', xy=(x, y + h), xytext=(2, -2),
                        textcoords='offset points', ha='left', va='top',
                        fontsize=10, color='white',
                        bbox={"facecolor": to_rgba(color, alpha=1.0),
                            "alpha": 1.0,
                            'pad': 2,
                            'edgecolor': 'none'})


def wandb_plot_text(generated: List[List[str]], target: List[List[str]], max_samples=None):
    assert len(generated) == len(target)
    if max_samples is not None and len(generated) > max_samples:
        generated = generated[:max_samples]
        target = target[:max_samples]
    columns = ["Sample", "Box ID", "Target", "Generated"]
    rows = []
    for i, (gen, tgt) in enumerate(zip(generated, target)):
        assert len(gen) == len(tgt)
        rows.extend([[i, j, t, g] for j, (t, g) in enumerate(zip(tgt, gen))])
    return wandb.Table(data=rows, columns=list(columns))

