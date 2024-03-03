from typing import List
import einops
import torch
from torchvision.ops import batched_nms, box_convert
from ensemble_boxes import weighted_boxes_fusion


def clip_bboxes(box_params):
    box_params = box_convert(box_params, 'cxcywh', 'xyxy')
    box_params = box_params.clamp(0., 1.)
    box_params = box_convert(box_params, 'xyxy', 'cxcywh')
    return box_params


def apply_top1_filtering(boxes: List[torch.Tensor]) -> List[torch.Tensor]:
    filtered_boxes = []
    for sample_boxes in boxes:
        labels = sample_boxes[:, 4]
        scores = sample_boxes[:, 5]
        unique_classes = torch.unique(labels)
        keep_inds = torch.stack([
            (torch.where(labels == cls, 1, 0) * scores).argmax()
            for cls in unique_classes
        ]) if len(unique_classes) > 0 else torch.zeros(0, dtype=torch.long)
        filtered_boxes.append(sample_boxes[keep_inds])
    return filtered_boxes


def apply_top1_with_box_fusion(boxes: List[torch.Tensor]) -> List[torch.Tensor]:
    filtered_boxes = []
    for sample_boxes in boxes:
        boxes = sample_boxes[:, :4]
        boxes_upper_left = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_lower_right = boxes[:, :2] + boxes[:, 2:] / 2
        labels = sample_boxes[:, 4]
        scores = sample_boxes[:, 5]
        unique_classes = torch.unique(labels)
        filtered_sample_boxes = []
        for c in unique_classes:
            cls_scores = scores[labels == c]
            if len(cls_scores) == 0:
                continue
            cls_boxes_upper_left = boxes_upper_left[labels == c]
            cls_boxes_lower_right = boxes_lower_right[labels == c]
            cls_fused_boxes = torch.cat([cls_boxes_upper_left.amin(dim=0), cls_boxes_lower_right.amax(dim=0)], dim=-1)
            wh = cls_fused_boxes[2:] - cls_fused_boxes[:2]
            cls_fused_boxes = torch.cat([cls_fused_boxes[:2] + wh / 2, wh], dim=-1)
            cls_top1_scores = cls_scores.amax()
            filtered_sample_boxes.append(torch.cat([cls_fused_boxes, c.unsqueeze(-1), cls_top1_scores.unsqueeze(-1)], dim=-1))
        filtered_boxes.append(torch.stack(filtered_sample_boxes) if len(filtered_sample_boxes) > 0 else torch.zeros(0, 6))
    return filtered_boxes


def apply_nms(predicted_boxes: List[torch.Tensor], iou_threshold: float):
    predicted_boxes_after_nms = []
    for sample_boxes in predicted_boxes:
        boxes_coords = box_convert(sample_boxes[:, 0:4], in_fmt='cxcywh', out_fmt='xyxy')
        cls_idxs = sample_boxes[:, 4]
        scores = sample_boxes[:, 5]
        nms_indices = batched_nms(boxes_coords, scores, cls_idxs, iou_threshold=iou_threshold)
        predicted_boxes_after_nms.append(sample_boxes[nms_indices, :])
    return predicted_boxes_after_nms

