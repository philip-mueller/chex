

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import repeat
import logging
from typing import Dict, List, Optional
import einops
from omegaconf import MISSING
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from metrics.detection_metrics import DynamicSetDetectionMetrics, filter_obj_threshold
from model.components.bbox_prediction import apply_nms, apply_top1_filtering, apply_top1_with_box_fusion, clip_bboxes
from model.components.classification_losses import classify_features
from model.detector.token_decoder_detector import TokenDetectorOutput

from util.model_utils import BaseModelOutput
from util.plot_utils import prepare_wandb_bbox_images
from util.train_utils import EvalConfig, Evaluator


log = logging.getLogger(__name__)

@dataclass
class PathologyDetectionOutput(BaseModelOutput):
    # (N x H x W)
    image: torch.Tensor = MISSING

    # List (N) of tensors (M_i x 6) in the (x_c, y_c, w, h, class_id, score) format
    pred_boxes: List[torch.Tensor] = MISSING
    # List (N) of tensors (M_i x 5) in the (x_c, y_c, w, h, class_id) format
    target_cls_boxes: Optional[List[torch.Tensor]] = None

@dataclass
class PathologyDetectionEvalConfig(EvalConfig):
    box_prompts: Dict[str, List[str]] = MISSING

    pos_prompts: Dict[str, List[str]] = MISSING
    neg_prompts: Dict[str, List[str]] = MISSING
    no_finding_prompts: List[str] = field(default_factory=list)

    classify_by_weights: bool = False
    classify_by_prompts: bool = True

    neg_prompt_mode: str = 'neg'  # neg, pos_centroid, no_finding
    normalize_classification: bool = True
    softmax_temperature: Optional[float] = None

    obj_threshold: Optional[float] = None
    box_scale_factors: Optional[Dict[str, float]] = None
    postprocess: Optional[str] = None

    clip_boxes: bool = True
    nms_threshold: float = 0.25
    
    skip_roi_pool_inference: bool = True


class PathologyDetectionEvaluator(Evaluator):
    def __init__(self, config: PathologyDetectionEvalConfig, model: 'ChEX', **kwargs):
        super().__init__(config, config_cls=PathologyDetectionEvalConfig, **kwargs)
        from src.model.chex import ChEX
        self.model: ChEX = model
        config = self.config

        assert self.dataset.has_class_bboxes, 'Dataset does not have class bboxes (has_class_bboxes = False in the config)'
        assert self.dataset.class_names is not None and len(self.dataset.class_names) > 0, 'Dataset does not have class names (missing class_names in the config)'
        self.class_names = self.dataset.class_names

        # Encode prompts
        # (C x d), (C)
        self.box_prompt_emb, self.box_prompt_mask = self.model.encode_prompts(self.config.box_prompts, self.class_names)
        
        # (C x d)
        self.pos_prompt_emb, self.neg_prompt_emb = self.model.encode_pos_neg_prompts(
            self.config.neg_prompt_mode, self.class_names,
            self.config.pos_prompts, self.config.neg_prompts, self.config.no_finding_prompts)

        self._register_metric(DynamicSetDetectionMetrics(
            self.class_names, 
            return_class_metrics=True, return_class_stats=False,
            return_classification_metrics=False,
            obj_threshold=self.config.obj_threshold,
            map_iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ap_iou_thresholds=[0.1, 0.3, 0.5]))

    def _predict(self, 
        x: torch.Tensor, 
        target_cls_boxes: Optional[List[torch.FloatTensor]] = None,
        **kwargs):

        d_prompt = self.box_prompt_emb.shape[-1]
    
        # ---> Detect
        detected_regions: TokenDetectorOutput = self.model.detect_prompts(
            x, self.box_prompt_emb.view(-1, d_prompt), self.box_prompt_mask.view(-1), 
            clip_boxes=self.config.clip_boxes, 
            skip_roi_pool=self.config.skip_roi_pool_inference,
            use_post_decoder=False)
        
        return detected_regions, x, target_cls_boxes

    def _postprocess(self, detected_regions: TokenDetectorOutput, x, target_cls_boxes, config: PathologyDetectionEvalConfig):
        N = detected_regions.box_features.shape[0]
        C = self.box_prompt_emb.shape[0]
        device = detected_regions.box_features.device
        R = detected_regions.multiboxes_features.shape[2]
        # (N x C x R x d)
        region_features = detected_regions.multiboxes_features
        region_boxes = detected_regions.multiboxes
        # (N x C x R)
        class_ids = einops.repeat(torch.arange(C, device=device, dtype=torch.long), 'c -> n c r', n=N, r=R)
        region_mask = torch.ones_like(class_ids, dtype=torch.bool)
        region_probs = detected_regions.multiboxes_weights if config.classify_by_weights else region_mask.to(dtype=region_features.dtype)
        
        # ---> Classify
        # (N x C x d)
        pos_prompt_emb = einops.repeat(self.pos_prompt_emb, 'c d -> n c r d', n=N, r=R)
        neg_prompt_emb = einops.repeat(self.neg_prompt_emb, 'c d -> n c r d', n=N, r=R)

        # (N x C) or (N x C x R)
        if config.classify_by_prompts:
            region_cls_probs, region_cls_mask = classify_features(
                region_features, pos_prompt_emb, neg_prompt_emb, 
                normalized=config.normalize_classification, 
                softmax=config.softmax_temperature is not None, temp=config.softmax_temperature if config.softmax_temperature is not None else 1.0)
            region_probs = region_probs * region_cls_probs
        
        if config.box_scale_factors is not None:
            # (C)
            scale_factors = torch.tensor([config.box_scale_factors[class_name] for class_name in self.class_names], device=device)
            region_boxes = region_boxes.clone()
            region_boxes[:, :, :, 2:4] *= scale_factors[None, :, None, None]

            if self.config.clip_boxes:
                region_boxes = clip_bboxes(region_boxes)

        # (N x C x 6) or (N x C x R x 6)
        boxes = torch.cat([region_boxes, class_ids[..., None].float(), region_probs[..., None]], dim=-1)
        # List (N) of tensors (M_i x 6)
        boxes = [sample_boxes[sample_mask] for sample_boxes, sample_mask in zip(boxes, region_mask)]
        if config.postprocess == 'nms':
            boxes = apply_nms(boxes, iou_threshold=config.nms_threshold)
        if config.postprocess == 'top1':
            boxes = apply_top1_filtering(boxes)
        elif config.postprocess == 'top1_boxfusion':
            boxes = apply_top1_with_box_fusion(boxes)

        return PathologyDetectionOutput(
            image=x,
            pred_boxes=boxes,
            target_cls_boxes=target_cls_boxes)

    def _update_metrics_with_output(self, output: PathologyDetectionOutput):
        self._update_metric(predicted_boxes=output.pred_boxes, target_boxes=output.target_cls_boxes)

    def _do_inference_for_optimization(self, predictions: list, config: PathologyDetectionEvalConfig):
        self.reset_metrics()
        for sample_pred in predictions:
            output = self._postprocess(*sample_pred, config=config)
            self._update_metrics_with_output(output)

        metrics = self._get_metric().compute_mAP()
        self.reset_metrics()
        return metrics
    
    def optimize_inference(self, predictions: list, optimize_scale_factor: bool = True, optimize_postprocess: bool = True):
        base_config = self.config
        best_config_overwrites = {}
        log.info('Optimizing inference for mAP...')

        base_config = self.config

        if optimize_scale_factor:
            log.info('Optimizing box scale factor...')
            # different evaluation datasets draw smaller/larger boxes so we scale the boxes based on the val dataset
            box_scale_sweep = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
            best_mAP, best_scale = defaultdict(float), defaultdict(lambda: 1.0)
            for box_scale in tqdm(box_scale_sweep):
                current_config = deepcopy(base_config)
                current_config.box_scale_factors = {class_name: box_scale for class_name in self.class_names}
                mAPs = self._do_inference_for_optimization(predictions, current_config)
                mAPs = {class_name: mAPs[f'AP/classes/mAP_{class_name}'] for class_name in self.class_names}
                for class_name in self.class_names:
                    if mAPs[class_name] >= best_mAP[class_name]:
                        best_mAP[class_name], best_scale[class_name] = mAPs[class_name], box_scale
            base_config = deepcopy(base_config)
            base_config.box_scale_factors = {class_name: best_scale[class_name] for class_name in self.class_names}

            best_config_overwrites = {
                **best_config_overwrites,
                'box_scale_factors': base_config.box_scale_factors,
            }
            log.info(f'Optimized box sclaes method: mAPs={best_mAP}')

        if optimize_postprocess:
            # Optimize hyperparams for mAP
            # - classify_by_weights, classify_by_prompts, classify_by_weights+classify_by_prompts+softmax_temperature
            best_mAP, best_config = 0.0, None
            softmax_temp_sweep = np.arange(0.05, 0.5, 0.05).tolist()
            sweep = [(True, False, None), (False, True, None)] + list(zip(repeat(True), repeat(True), softmax_temp_sweep))
            for by_weights, by_prompts, temp in tqdm(sweep):
                current_config = deepcopy(base_config)
                current_config.classify_by_weights = by_weights
                current_config.classify_by_prompts = by_prompts
                current_config.softmax_temperature = temp
                mAP = self._do_inference_for_optimization(predictions, current_config)['AP/mAP']
                if mAP >= best_mAP:
                    best_mAP, best_config = mAP, current_config
            base_config = best_config
            log.info(f'Optimized classification method: mAP={best_mAP}')

            if base_config.postprocess == 'nms':
                best_map, best_config = 0.0, None
                nms_thres_sweep = [0.05, 0.1, 0.25, 0.5]
            
                for nms_thres in tqdm(nms_thres_sweep):
                    current_config = deepcopy(base_config)
                    current_config.nms_threshold = nms_thres
                    mAP = self._do_inference_for_optimization(predictions, current_config)['AP/mAP']
                    if mAP >= best_map:
                        best_map, best_config = mAP, current_config
                base_config = best_config
                log.info(f'Optimized nms: mAP={best_map}')

            best_config_overwrites = {
                **best_config_overwrites,
                'classify_by_weights': best_config.classify_by_weights, 
                'classify_by_prompts': best_config.classify_by_prompts,
                'softmax_temperature': best_config.softmax_temperature,
                'nms_threshold': best_config.nms_threshold,
            }

        log.info(f'Best config: {best_config_overwrites}')     
        return base_config


    def plot(self, output: PathologyDetectionOutput, max_samples: int, target_dir: str, plot_local):
        pred_boxes = filter_obj_threshold(output.pred_boxes, self.config.obj_threshold) if self.config.obj_threshold is not None else output.pred_boxes
        wandb_boxes = prepare_wandb_bbox_images(
            images=output.image, 
            preds=pred_boxes, targets=output.target_cls_boxes, 
            class_names=self.class_names, one_box_per_class=False,
            max_samples=max_samples)
        return {'patho_boxes': wandb_boxes}
    