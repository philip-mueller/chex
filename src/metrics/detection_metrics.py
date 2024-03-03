
from typing import Dict, List, Optional, Sequence, Tuple, Union
import einops
from torch import Tensor, nn
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification.precision_recall import MultilabelRecall, MultilabelPrecision
from torchmetrics.classification.f_beta import MulticlassF1Score
from torchmetrics.classification.auroc import MultilabelAUROC
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou, box_convert
import numpy as np
from mean_average_precision import MeanAveragePrecision2d
from model.components.bbox_prediction import clip_bboxes

from util.data_utils import to_device

class SentenceDetectionMetrics(nn.Module):
    def __init__(self, 
                 map_iou_thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
                 ap_iou_thresholds: Tuple = (0.5, 0.75),
                 obj_threshold: Optional[float] = None,
                 iou_resolution: Optional[int] = 1024,
                 mIoU_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5), # (0.55, 0.6, 0.65, 0.7, 0.75), #(0.1, 0.2, 0.3, 0.4, 0.5),
                 device='cpu',
                 mask_vs_box=False
) -> None:
        super().__init__()
        self.iou_resolution = iou_resolution
        self.AP = BoxAPMetric(class_names=['sentence'],
                              return_classes=False,
                              map_iou_thresholds=map_iou_thresholds, ap_iou_thresholds=ap_iou_thresholds)
        self.obj_threshold = obj_threshold
        # self.register_buffer('greedy_iou', torch.tensor(0.))
        # self.register_buffer('N', torch.tensor(0.0))
        
        self.iou = BoxMaskIoUMetric(obj_threshold, iou_resolution)
        self.mIoU_thresholds = mIoU_thresholds
        self.mIoUs = nn.ModuleList([BoxMaskIoUMetric(threshold, iou_resolution) for threshold in mIoU_thresholds])

        self.mask_vs_box = mask_vs_box
        if mask_vs_box:
            self.mask_mIoUs = nn.ModuleList([SegMaskIoUMetric(C=1) for threshold in mIoU_thresholds])

        self.device = device
        self.to(device)

        
    @torch.inference_mode()
    def update(self, 
               predicted_boxes: List[torch.Tensor], 
               target_boxes: List[List[torch.Tensor]],
               predicted_mask_probs: Optional[List[torch.Tensor]] = None):
        """
        :param predicted_boxes: List (N) of tensors (S_i x M_pred x 5) in the (x_c, y_c, w, h, obj_prob) format
        :param target_boxes: List (N) of lists (S_i) of tensors (M_is x 5) in the (x_c, y_c, w, h) format
        :param predicted_masks:  List (N) of tensors (S_i x H x W)
        """
        N = len(predicted_boxes)
        predicted_boxes = to_device(predicted_boxes, self.device)
        target_boxes = to_device(target_boxes, self.device)
        assert len(predicted_boxes) == len(target_boxes), 'Number of predicted boxes and target boxes must be the same'
        assert all([len(pred_boxes) == len(targ_boxes) for pred_boxes, targ_boxes in zip(predicted_boxes, target_boxes)]), 'Number of predicted boxes and target boxes must be the same'
        # Every sentence treated as independent sample
        # List of length (N * S_i) of tensors (M_pred x 5)
        flattened_pred_boxes = [pred_boxes for sample_pred_boxes in predicted_boxes for pred_boxes in sample_pred_boxes]
        flattened_target_boxes = [targ_boxes for sample_targ_boxes in target_boxes for targ_boxes in sample_targ_boxes]

        # ------- AP -------
        cls_id = torch.tensor(0, device=self.device, dtype=torch.long)
        pred_boxes_with_cls = [
            torch.cat([pred_boxes[:, :4], cls_id[None, None].expand(pred_boxes.shape[0], 1), pred_boxes[:, 4, None]], dim=1)
            for pred_boxes in flattened_pred_boxes
        ]
        target_boxes_with_cls = [
            torch.cat([targ_boxes[:, :4], cls_id[None, None].expand(targ_boxes.shape[0], 1)], dim=1)
            for targ_boxes in flattened_target_boxes
        ]
        self.AP.update(pred_boxes_with_cls, target_boxes_with_cls)
        filtered_pred_boxes_with_cls = filter_obj_threshold(pred_boxes_with_cls, self.obj_threshold) if self.obj_threshold is not None else pred_boxes_with_cls
        
        # ----> IoU and mIoU
        self.iou.update(filtered_pred_boxes_with_cls, target_boxes_with_cls)
        for mIoU, t in zip(self.mIoUs, self.mIoU_thresholds):
            mIoU.update(filter_obj_threshold(pred_boxes_with_cls, t), target_boxes_with_cls)

        if self.mask_vs_box:
            assert predicted_mask_probs is not None
            predicted_mask_probs = torch.stack([mask_probs for sample_mask_probs in predicted_mask_probs for mask_probs in sample_mask_probs]).unsqueeze(1)
            for mIoU, t in zip(self.mask_mIoUs, self.mIoU_thresholds):
                mIoU.update(predicted_mask_probs > t, target_boxes_with_cls)

    def reset(self):
        self.AP.reset()
        self.iou.reset()
        for mIoU in self.mIoUs:
            mIoU.reset()
        if self.mask_vs_box:
            self.mask_IoU.reset()
            for mIoU in self.mask_mIoUs:
                mIoU.reset()

    @torch.inference_mode()
    def compute(self) -> dict:
        iou = self.iou.compute()
        miou = torch.stack([mIoU_metric.compute() for mIoU_metric in self.mIoUs])
        metrics = {
            **{k: v.cpu() for k, v in self.AP.compute().items()},
            'IoU': iou.cpu(),
            'mIoU': miou.mean().cpu(),
            'bIoU': miou.amax().cpu(),
        }
        if self.mask_vs_box:
            mask_miou = torch.stack([mIoU_metric.compute() for mIoU_metric in self.mask_mIoUs])
            metrics['mask_mIoU'] = mask_miou.mean().cpu()
            metrics['mask_bIoU'] = mask_miou.amax().cpu()
            metrics.update({f'mask_mIoU@{t}': mIoU.cpu() for t, mIoU in zip(self.mIoU_thresholds, mask_miou)})
        return metrics
    
    @torch.inference_mode()
    def compute_mAP(self) -> dict:
        return {k: v.cpu() for k, v in self.AP.compute().items()}
    

class BoxMaskIoUMetric(nn.Module):
    def __init__(self, obj_threshold, iou_resolution) -> None:
        super().__init__()
        self.iou_resolution = iou_resolution
        self.obj_threshold = obj_threshold

        self.register_buffer('intersection', torch.tensor(0.))
        self.register_buffer('union', torch.tensor(0.))

    @torch.inference_mode()
    def update(self, 
               predicted_boxes: List[torch.Tensor], 
               target_boxes: List[List[torch.Tensor]]):
        
        filtered_pred_boxes = filter_obj_threshold(predicted_boxes, self.obj_threshold)
        # ----> mIoU
        inter, union = bboxes_mask_miou(filtered_pred_boxes, target_boxes, self.iou_resolution, C=1)
        self.intersection += inter.squeeze()
        self.union += union.squeeze()

    def reset(self):
        self.intersection.zero_()
        self.union.zero_()
        
    @torch.inference_mode()
    def compute(self) -> dict:
        return self.intersection / self.union.clamp(min=1)
    
class SegMaskIoUMetric(nn.Module):
    def __init__(self, C) -> None:
        super().__init__()

        self.register_buffer('intersection', torch.zeros(C))
        self.register_buffer('union', torch.zeros(C))

    @torch.inference_mode()
    def update(self, 
               predicted_mask_probs: torch.FloatTensor,
               target_boxes: List[List[torch.Tensor]]):
        
        inter, union = segmask_to_bboxes_mask_miou(predicted_mask_probs, target_boxes)
        self.intersection += inter
        self.union += union

    def reset(self):
        self.intersection.zero_()
        self.union.zero_()
        
    @torch.inference_mode()
    def compute(self) -> dict:
        return (self.intersection / self.union.clamp(min=1)).mean()

    

def sentences_to_ids(cls_box_sentences: List[List[str]], device) -> List[torch.LongTensor]:
    sentence_ids = []
    start_id = 0
    for sample_sentences in cls_box_sentences:
        all_sentences = list(set(sample_sentences))
        sample_sentence_ids = torch.tensor([all_sentences.index(s) + start_id for s in sample_sentences],
                                           device=device, dtype=torch.long)
        sentence_ids.append(sample_sentence_ids)
        start_id += len(all_sentences)

    return sentence_ids

def bboxes_mask_miou(bboxes1: List[torch.Tensor], bboxes2: List[torch.Tensor], img_size: int, C: int) -> torch.Tensor:
    masks1 = bboxes_to_mask(bboxes1, img_size, C)
    masks2 = bboxes_to_mask(bboxes2, img_size, C)
    assert masks1.shape == masks2.shape, 'Masks must have the same shape'

    # (C) <- (C x N x H x W)
    intersection = (masks1 & masks2).float().sum(dim=(1, 2, 3))
    union = (masks1 | masks2).float().sum(dim=(1, 2, 3))

    return intersection, union


def segmask_to_bboxes_mask_miou(masks: torch.BoolTensor, bboxes: List[torch.Tensor]) -> torch.Tensor:
    N, C, H, W = masks.shape
    assert H == W, 'Masks must be square'
    img_size = H
    masks = einops.rearrange(masks, 'n c h w -> c n h w')
    masks_bbox = bboxes_to_mask(bboxes, img_size, C)
    assert masks.shape == masks_bbox.shape, 'Masks must have the same shape'

    # (C) <- (C x N x H x W)
    intersection = (masks & masks_bbox).float().sum(dim=(1, 2, 3))
    union = (masks | masks_bbox).float().sum(dim=(1, 2, 3))

    return intersection, union

@torch.jit.script
def bboxes_to_mask(bboxes: List[torch.Tensor], img_size: int, C: int) -> torch.BoolTensor:
    """
    :param bboxes: List (N) of tensors (M_i x 5) in the (x_c, y_c, w, h, cls_id) format
    :return mask: Bool tensor (C x N x H x W)
    """
    N = len(bboxes)
    mask = torch.zeros((C, N, img_size, img_size), dtype=torch.bool)
    for i, boxes in enumerate(bboxes):
        for box in boxes:
            cls_id = box[4].int()
            box = img_size * box[:4]
            wh = box[2:4]
            center = box[:2]
            x1y1 = (center - wh / 2).int().clamp_max(img_size)
            x2y2 = (center + wh / 2).int().clamp_max(img_size)
            x1, y1 = x1y1[0], x1y1[1]
            x2, y2 = x2y2[0], x2y2[1]
            mask[cls_id, i, y1:y2, x1:x2] = True
    return mask


class FixedSetDetectionMetrics(nn.Module):
    def __init__(self, 
                 class_names: List[str], 
                 return_class_metrics: bool = False, return_class_stats: bool = False, mask_prediction_metrics: bool = False,
                 map_iou_thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
                 ap_iou_thresholds: Tuple = (0.5, 0.75),
                 device='cpu') -> None:
        super().__init__()
        self.class_names = [cls_name.replace(' ', '_').replace('/', '-') for cls_name in class_names]
        self.stats = BoxStatistics(class_names, return_classes=return_class_stats)
        self.IoU = BoxIoUMetric(class_names, return_classes=return_class_metrics)
        self.AP = BoxAPMetric(class_names, 
                              return_classes=False,
                              map_iou_thresholds=map_iou_thresholds, ap_iou_thresholds=ap_iou_thresholds)

        self.mask_prediction_metrics = mask_prediction_metrics
        if mask_prediction_metrics:
            self.mask_prec_recall = MetricCollection({'presence_mask/prec': MultilabelPrecision(len(class_names)), 'presence_mask/recall': MultilabelRecall(len(class_names))})
            self.mask_auroc = MultilabelAUROC(len(class_names))

        self.device = device
        self.to(device)

    @torch.inference_mode()
    def update(self, 
               predicted_boxes: torch.Tensor, 
               target_boxes: torch.Tensor, 
               predicted_mask_probs: Optional[torch.Tensor] = None,
               predicted_masks: Optional[torch.Tensor] = None, 
               target_masks: Optional[torch.Tensor] = None):
        
        predicted_boxes = to_device(predicted_boxes, self.device)
        target_boxes = to_device(target_boxes, self.device)
        predicted_mask_probs = to_device(predicted_mask_probs, self.device)
        predicted_masks = to_device(predicted_masks, self.device)
        target_masks = to_device(target_masks, self.device)

        self.stats.update(predicted_boxes, mask=predicted_masks)
        self.IoU.update(predicted_boxes, target_boxes, target_masks=target_masks)
        
        predictions_for_ap = convert_fixed_set_to_dynamic_set_boxes(predicted_boxes, box_probs=predicted_mask_probs, mask=target_masks)
        targets_for_ap = convert_fixed_set_to_dynamic_set_boxes(target_boxes, mask=target_masks, with_probs=False)
        self.AP.update(predictions_for_ap, targets_for_ap)

        if self.mask_prediction_metrics:
            assert predicted_masks is not None and target_masks is not None
            self.mask_prec_recall.update(predicted_masks, target_masks)
            self.mask_auroc.update(predicted_mask_probs, target_masks)


    def reset(self):
        self.stats.reset()
        self.IoU.reset()
        self.AP.reset()
        if self.mask_prediction_metrics:
            self.mask_prec_recall.reset()
            self.mask_auroc.reset()

    @torch.inference_mode()
    def compute(self) -> dict:
        metrics = {
            **{k: v.cpu() for k, v in self.stats.compute().items()},
            **{k: v.cpu() for k, v in self.IoU.compute().items()},
            **{k: v.cpu() for k, v in self.AP.compute().items()},
        }
        if self.mask_prediction_metrics:
            metrics.update({k: v.cpu() for k, v in self.mask_prec_recall.compute().items()})
            metrics['presence_mask/auroc'] = self.mask_auroc.compute().cpu()
        return metrics
    

class DynamicSetDetectionMetrics(nn.Module):
    def __init__(self, 
                 class_names: List[str], 
                 return_class_metrics: bool = False, return_class_stats: bool = False, 
                 return_classification_metrics: bool = False,
                 obj_threshold: Optional[float] = None,
                 obj_cls_thresholds: Optional[Dict[str, float]] = None,
                 map_iou_thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
                 ap_iou_thresholds: Tuple = (0.5, 0.75),
                 device='cpu') -> None:
        super().__init__()
        self.original_class_names = class_names
        self.class_names = [cls_name.replace(' ', '_').replace('/', '-') for cls_name in class_names]
        self.stats = BoxStatistics(class_names, return_classes=return_class_stats) if return_class_stats else None
        self.AP = BoxAPMetric(class_names, 
                              return_classes=return_class_metrics,
                              map_iou_thresholds=map_iou_thresholds, ap_iou_thresholds=ap_iou_thresholds)

        self.return_classification_metrics = return_classification_metrics
        if return_classification_metrics:
            self.f1 = MulticlassF1Score(len(class_names))
            self.auroc = MultilabelAUROC(len(class_names))

        assert obj_cls_thresholds is None or obj_threshold is None, 'obj_cls_thresholds and obj_threshold cannot be used together'
        if obj_cls_thresholds is not None:
            self.obj_cls_thresholds = obj_cls_thresholds
        elif obj_threshold is not None:
            self.obj_cls_thresholds = obj_threshold
        else:
            self.obj_cls_thresholds = None
        self.device = device
        self.to(device)

    @torch.inference_mode()
    def update(self, 
               predicted_boxes: List[torch.Tensor],
               target_boxes: List[torch.Tensor]):
        predicted_boxes = to_device(predicted_boxes, self.device)
        target_boxes = to_device(target_boxes, self.device)
        
        if self.stats is not None:
            self.stats.update(predicted_boxes)
        self.AP.update(predicted_boxes, target_boxes)

        filtered_predicted_boxes = filter_obj_threshold(predicted_boxes, self.obj_cls_thresholds, self.original_class_names)

        if self.return_classification_metrics:
            device = self.device
            N, C = len(filtered_predicted_boxes), len(self.class_names)
            # (N x C)
            predicted_classes = torch.zeros((N, C), device=device, dtype=torch.long)
            predicted_class_probs = torch.zeros((N, C), device=device, dtype=torch.float32)
            target_classes = torch.zeros((N, C), device=device, dtype=torch.long)

            class_ids = torch.arange(C, device=device, dtype=torch.long)

            for i, (sample_pred_boxes, sample_target_boxes) in enumerate(zip(filtered_predicted_boxes, target_boxes)):
                M_pred = sample_pred_boxes.shape[0]
                if M_pred > 0:
                    # (M_pred)
                    box_pred_classes = sample_pred_boxes[:, 4].long()
                    # (C x M_pred)
                    pred_class_mask = box_pred_classes.unsqueeze(0) == class_ids.unsqueeze(1)
                    # (C x M_pred)
                    pred_class_probs = sample_pred_boxes[:, 5].unsqueeze(0) * pred_class_mask

                    predicted_classes[i] = pred_class_mask.any(dim=1).long()
                    predicted_class_probs[i] = pred_class_probs.amax(dim=1)

                M_target = sample_target_boxes.shape[0]
                if M_target > 0:
                    # (M_target)
                    box_target_classes = sample_target_boxes[:, 4].long()
                    # (C x M_target)
                    target_class_mask = box_target_classes.unsqueeze(0) == class_ids.unsqueeze(1)
                    target_classes[i] = target_class_mask.any(dim=1).long()
            
            self.f1.update(predicted_classes, target_classes)
            self.auroc.update(predicted_class_probs, target_classes)

    def reset(self):
        if self.stats is not None:
            self.stats.reset()
        self.AP.reset()
        if self.return_classification_metrics:
            self.f1.reset()
            self.auroc.reset()

    @torch.inference_mode()
    def compute(self) -> dict:
        results = {
            **{k: v.cpu() for k, v in self.AP.compute().items()},
        }
        if self.stats is not None:
            results.update(**{k: v.cpu() for k, v in self.stats.compute().items()})
        if self.return_classification_metrics:
            results['classification/f1'] = self.f1.compute().cpu()
            results['classification/auroc'] = self.auroc.compute().cpu()
        return results
    
    @torch.inference_mode()
    def compute_mAP(self) -> dict:
        return {k: v.cpu() for k, v in self.AP.compute().items()}
    

def convert_fixed_set_to_dynamic_set_boxes(
        boxes: torch.Tensor, box_probs: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, 
        convert_to_format: Optional[str] = None, with_probs: bool = True, to_np: bool = False) -> List[torch.Tensor]:
    """
    :param boxes: (N, C, 4) in format (x_c, y_c, w, h)
    :param box_probs: (N, C)
    :param mask: (N, C)
    """
    if convert_to_format is not None:
        boxes = box_convert(boxes, in_fmt='cxcywh', out_fmt=convert_to_format)

    N, C, _ = boxes.shape
    class_ids = einops.repeat(torch.arange(C, device=boxes.device, dtype=boxes.dtype), 'c -> n c', n=N)
    if mask is None:
        mask = torch.ones_like(class_ids, dtype=torch.bool)
    if with_probs and box_probs is None:
        box_probs = mask.float()
    # (N, C, 6) in format (<convert_to_format>, class_id, box_prob) or (<convert_to_format>, class_id)
    boxes = torch.cat([boxes, class_ids.unsqueeze(-1), box_probs.unsqueeze(-1)], dim=-1) if with_probs else torch.cat([boxes, class_ids.unsqueeze(-1)], dim=-1)

    if to_np:
        boxes = boxes.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

    data = [sample_boxes[sample_mask] for sample_boxes, sample_mask in zip(boxes, mask)]
    return data


def convert_box_format(boxes: List[Tensor], convert_to_format: str, to_np: bool = False) -> List[Tensor]:
    boxes = [
        torch.cat([box_convert(sample_boxes[:, :4], in_fmt='cxcywh', out_fmt=convert_to_format), sample_boxes[:, 4:]], dim=-1)
         for sample_boxes in boxes]
    if to_np:
        boxes = [sample_boxes.detach().cpu().numpy() for sample_boxes in boxes]
    return boxes


def filter_obj_threshold(boxes: List[Tensor], obj_threshold: Union[float, Dict[str, float]], class_names: List[str]=None) -> List[Tensor]:
    if obj_threshold is None or len(boxes) == 0:
        return boxes
    elif isinstance(obj_threshold, float):
        return [sample_boxes[sample_boxes[:, 5] > obj_threshold] for sample_boxes in boxes]
    else:
        assert class_names is not None, 'class_names must be provided if obj_threshold is a dict'
        # (C)
        obj_thresholds_by_cls_index = torch.tensor([obj_threshold[cls_name] for cls_name in class_names], device=boxes[0].device, dtype=boxes[0].dtype)
        # List of length (N) of tensors (M_i)
        box_thresholds = [obj_thresholds_by_cls_index[sample_boxes[:, 4].long()] for sample_boxes in boxes]
        return [sample_boxes[sample_boxes[:, 5] > box_threshold] for sample_boxes, box_threshold in zip(boxes, box_thresholds)]


class BoxAPMetric:
    """
    Compute mean Average Precision for bounding boxes with
    https://github.com/bes-dev/mean_average_precision

    For COCO, select iou_thresholds = np.arange(0.5, 1.0, 0.05)
    For Pascal VOC, select iou_thresholds = 0.5
    """
    def __init__(self, class_names: int, map_iou_thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
                 ap_iou_thresholds: Tuple = (0.5, 0.75), return_classes: bool = False):
        super(BoxAPMetric, self).__init__()
        self.iou_thresholds = np.array(map_iou_thresholds)
        self.extra_reported_thresholds = []
        for extra_thres in ap_iou_thresholds:
            found_close = False
            for thres in self.iou_thresholds:
                if np.isclose(thres, extra_thres):
                    self.extra_reported_thresholds.append(thres)
                    found_close = True
            if not found_close:
                raise ValueError(f'{extra_thres} not found in {self.iou_thresholds}')
        self.metric = MeanAveragePrecision2d(len(class_names))
        self.class_names = class_names
        self.return_classes = return_classes

    def reset(self):
        self.metric.reset()

    def update(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        Add a batch of predictions and targets to the metric.
        N samples with each M bounding boxes of C classes.

        src.model.model_interface.ObjectDetectorPrediction.box_prediction_hard
        has the correct format for predictions.

        :param predictions: List of N predictions, each a tensor of shape (M x 6)
                            (x_c, y_c, w, h, class_id, confidence)
        :param targets: List of N targets, each a tensor of shape (M x 5)
                        (x_c, y_c, w, h, class_id)
        """
        for predicted, target in zip(predictions, targets):
            predicted_np = predicted.detach().cpu().numpy().copy()
            target_np = target.detach().cpu().numpy().copy()
            assert predicted_np.shape[-1] == 6, predicted_np.shape
            assert target_np.shape[-1] == 5, target_np.shape

            # Convert from [xc, yc, w, h, class_id, confidence]
            # to [xmin, ymin, xmax, ymax, class_id, confidence]
            preds = np.zeros((len(predicted_np), 6))
            preds[:, 0:2] = predicted_np[:, :2] - predicted_np[:, 2:4] / 2
            preds[:, 2:4] = predicted_np[:, :2] + predicted_np[:, 2:4] / 2
            preds[:, 4:6] = predicted_np[:, 4:6]

            # Convert from [xc, yc, w, h, class_id]
            # to [xmin, ymin, xmax, ymax, class_id, difficult]
            gt = np.zeros((len(target_np), 7))
            gt[:, 0:2] = target_np[:, :2] - target_np[:, 2:4] / 2
            gt[:, 2:4] = target_np[:, :2] + target_np[:, 2:4] / 2
            gt[:, 4] = target_np[:, 4]

            # --- correction as the metric implementation assumes pixels and therefore computes width/height offset by 1. ---
            preds[:, 0:4] *= 1000 
            preds[:, 2:4] -= 1.0
            gt[:, 0:4] *= 1000 
            gt[:, 2:4] -= 1.0
            # --- end correction ---

            self.metric.add(preds, gt)

    def compute(self) -> Dict[str, Tensor]:
        computed_metrics = self.metric.value(iou_thresholds=self.iou_thresholds,
                                             mpolicy="soft",
                                             recall_thresholds=np.arange(0., 1.01, 0.01))
        metrics = {'AP/mAP': torch.tensor(computed_metrics['mAP'])}

        if self.return_classes:
            for c, class_name in enumerate(self.class_names):
                metrics[f'AP/classes/mAP_{class_name}'] = torch.tensor(np.mean([computed_metrics[t][c]['ap'] for t in self.iou_thresholds]))
                #for t in self.extra_reported_thresholds:
                #    metrics[f'mAP@{t}_classes/{class_name}'] = computed_metrics[t][c]['ap']
            
        if self.extra_reported_thresholds is not None:
            for t in self.extra_reported_thresholds:
                metrics[f'AP/AP@{t}'] = torch.tensor(np.mean([computed_metrics[t][c]['ap'] for c in range(len(self.class_names))]))

        return metrics


class BoxIoUMetric(Metric):
    def __init__(self, class_names: List[str], return_classes: bool = False):
        super(BoxIoUMetric, self).__init__()
        self.class_names = class_names
        self.return_classes = return_classes
        self.add_state('count', torch.tensor(0.))
        self.add_state('samplemircoIoU_sum', torch.tensor(0.))  # (1)
        self.add_state('cls_intersection_sum', torch.zeros(len(class_names)))  # (C)
        self.add_state('cls_union_sum', torch.zeros(len(class_names)))  # (C)

    def update(self, predictions: Tensor, targets: Tensor, target_masks: Optional[Tensor] = None):
        """
        :param predictions: (N x M x 4) tensor of predicted boxes in (x_c, y_c, w, h) format
        """
        assert predictions.shape == targets.shape, f'{predictions.shape} != {targets.shape}'
        assert predictions.ndim in [2, 3], f'predictions.ndim == {predictions.ndim}'
        if predictions.ndim == 2:
            predictions = predictions.unsqueeze(1)
            targets = targets.unsqueeze(1)

        self.count += len(predictions)

        # Compute intersections and unions
        # (N x A)
        intersection_area, union_area = batched_box_intersection_and_union(
            predictions, targets, box_mask_2=target_masks)
        # For sample micro IoU
        samples_micro_ious = intersection_area.sum(1) / union_area.sum(1).clamp_min(1e-7) # (N)
        self.samplemircoIoU_sum += samples_micro_ious.sum(0)  # (1)
        # For class and macro IoU
        self.cls_intersection_sum += intersection_area.sum(0)  # (C)
        self.cls_union_sum += union_area.sum(0)  # (C)
    
    def compute(self) -> Dict[str, Tensor]:
        sample_micro_iou = self.samplemircoIoU_sum / self.count  # (1)
        class_ious = self.cls_intersection_sum / self.cls_union_sum.clamp_min(1e-7)  # (C)
        macro_iou = class_ious.mean()

        results = {
            'IoU/sample_micro': sample_micro_iou,
            'IoU/macro': macro_iou,
        }
        if self.return_classes:
            for c, class_name in enumerate(self.class_names):
                results[f'IoU/classes/{class_name}'] = class_ious[c]

        return results


class BoxStatistics(Metric):
    def __init__(self, class_names: Optional[List[str]] = None, return_classes: bool = False) -> None:
        super().__init__()
        self.add_state('count', torch.tensor(0.))
        self.add_state('box_count', torch.tensor(0.))
        self.add_state('box_area_sum', torch.tensor(0.))
        self.add_state('box_area_max', torch.tensor(-torch.inf))
        self.add_state('box_area_min', torch.tensor(torch.inf))
        
        self.return_classes = return_classes
        if return_classes:
            assert class_names is not None
            self.class_names = class_names
            self.add_state('cls_box_count', torch.zeros(len(class_names)))
            self.add_state('cls_box_area_sum', torch.zeros(len(class_names)))
            self.add_state('cls_box_area_max', torch.full((len(class_names),), -torch.inf))
            self.add_state('cls_box_area_min', torch.full((len(class_names),), torch.inf))
    
    def update(self, predictions: Union[List[torch.Tensor], Tensor], mask: Optional[Tensor] = None):
        """
        :param predictions: one of:
            - list of N tensors each of shape (M_i x 6) in format (x_c, y_c, w, h, class_id, confidence)
            - tensor of shape (N x M x 4) in format (x_c, y_c, w, h), if return_classes is True then M is interpreted as the class dimension
        :param mask: optional mask of shape (N x M) to filter out predictions
        """
        self.count += len(predictions)

        if not torch.is_tensor(predictions):
            # (sum(M_i) x 6)
            all_predictions = torch.cat(predictions, dim=0)
            assert mask is None

            if self.return_classes:
                # (C)
                class_ids = torch.arange(len(self.class_names), device=all_predictions.device, dtype=all_predictions.dtype)
                # (sum(M_i))
                box_classes = all_predictions[:, 4]
                # (C x sum(M_i))
                class_masks = box_classes[None, :] == class_ids[:, None]
                # list with C elements, each of shape (n_c x 6)
                predictions_per_class: List[torch.Tensor] = [all_predictions[class_mask] for class_mask in class_masks]
        else:
            if self.return_classes:
                N, M, _ = predictions.shape
                if mask is not None:
                    predictions_per_class: List[torch.Tensor] = [class_pred[class_maks] for class_pred, class_maks in zip(predictions.unbind(dim=1), mask.unbind(dim=1))]
                else:
                    predictions_per_class: List[torch.Tensor] = list(predictions.unbind(dim=1))
            else:
                if mask is not None:
                    all_predictions = predictions[mask]
                else:
                    all_predictions = predictions.flatten(0, -2)

        # micro (ignore classes)
        if all_predictions.shape[0] > 0:
            self.box_count += all_predictions.shape[0]
            box_areas = all_predictions[:, 2] * all_predictions[:, 3]
            self.box_area_sum += box_areas.sum()
            self.box_area_max = torch.max(self.box_area_max, box_areas.max())
            self.box_area_min = torch.min(self.box_area_min, box_areas.min())        
        
        # per class
        if self.return_classes:
            for c, cls_predictions in enumerate(predictions_per_class):
                self.cls_box_count[c] += len(cls_predictions)
                cls_box_areas = cls_predictions[:, 2] * cls_predictions[:, 3]
                if len(cls_box_areas) == 0:
                    continue
                self.cls_box_area_sum[c] += cls_box_areas.sum()
                self.cls_box_area_max[c] = torch.max(self.cls_box_area_max[c], cls_box_areas.max())
                self.cls_box_area_min[c] = torch.min(self.cls_box_area_min[c], cls_box_areas.min())
            

    def compute(self):
        avg_box_count = self.box_count / self.count
        box_area_mean = self.box_area_sum / self.box_count
        results = {
                'boxstat/count': avg_box_count,
                'boxstat/area_mean': box_area_mean,
                'boxstat/area_max': self.box_area_max,
                'boxstat/area_min': self.box_area_min,
            }

        if self.return_classes:
            cls_avg_box_count = self.cls_box_count / self.count
            cls_box_area_mean = self.cls_box_area_sum / self.cls_box_count

            for c, cls_name in enumerate(self.class_names):
                results.update({
                    f'boxstat/classes/count/{cls_name}': cls_avg_box_count[c],
                    f'boxstat/classes/area_mean/{cls_name}': cls_box_area_mean[c],
                    f'boxstat/classes/area_max/{cls_name}': self.cls_box_area_max[c],
                    f'boxstat/classes/area_min/{cls_name}': self.cls_box_area_min[c],
                })
        return results

@torch.jit.script
def batched_box_intersection_and_union(boxes_1, boxes_2, box_mask_1: Optional[Tensor]=None, box_mask_2: Optional[Tensor]=None):
    """
    :param boxes_1: (... x 4) in the (x_c, y_c, w, h) format
    :param boxes_2: (... x 4) in the (x_c, y_c, w, h) format
    :param box_mask_1: (...)
    :param box_mask_2: (...)
    :return (...)
    """
    boxes_1 = clip_bboxes(boxes_1)
    boxes_2 = clip_bboxes(boxes_2)

    wh_1 = boxes_1[..., 2:4]
    areas_1 = wh_1[..., 0] * wh_1[..., 1]  # (...)
    x1y1_1 = boxes_1[..., :2] - 0.5 * wh_1  # (... x 2)
    x2y2_1 = boxes_1[..., :2] + 0.5 * wh_1  # (... x 2)
    if box_mask_1 is not None:
        areas_1 = box_mask_1 * areas_1

    wh_2 = boxes_2[..., 2:4]
    areas_2 = wh_2[..., 0] * wh_2[..., 1]  # (...)
    x1y1_2 = boxes_2[..., :2] - 0.5 * wh_2 # (... x 2)
    x2y2_2 = boxes_2[..., :2] + 0.5 * wh_2  # (... x 2)
    if box_mask_2 is not None:
        areas_2 = box_mask_2 * areas_2

    xx1yy1 = torch.maximum(x1y1_1, x1y1_2)  # (... x 2)
    xx2yy2 = torch.minimum(x2y2_1, x2y2_2)  # (... x 2)
    intersection_wh = (xx2yy2 - xx1yy1).clamp_min(0.)  # (... x 2)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  # (...)
    if box_mask_1 is not None:
        intersection_area = box_mask_1 * intersection_area
    if box_mask_2 is not None:
        intersection_area = box_mask_2 * intersection_area

    union_area = areas_1 + areas_2 - intersection_area  # (...)
    return intersection_area, union_area  # (...)
    

@torch.jit.script
def batched_box_iou(boxes_1, boxes_2, box_mask_1: Optional[Tensor]=None, box_mask_2: Optional[Tensor]=None):
    intersection_area, union_area = batched_box_intersection_and_union(boxes_1, boxes_2, box_mask_1, box_mask_2)
    return intersection_area / union_area.clamp_min(1e-7)  # (...)