from dataclasses import dataclass, field
import logging
from typing import Optional
import einops
from omegaconf import MISSING
from model.components.contrastive_losses import pathology_contrastive_loss
import torch.nn.functional as F
from torch import BoolTensor, nn

import torch
from model.components.bbox_losses import bbox_giou_ccost, bbox_l1_ccost, match_multiregions
from model.components.classification_losses import classify_features, get_focal_loss

from model.detector.token_decoder_detector import TokenDetectorOutput
from model.img_encoder import ImageEncoderOutput
from model.supervisors.utils import subsample_classes
from util.model_utils import prepare_config

log = logging.getLogger(__name__)


@dataclass
class PathologyTokenConfig:
    # --- Loss: Pathology bounding box (patho-det) ---
    use_patho_detect: bool = False
    coeff_patho_detect: float = 0.0
    subsample_patho_boxes: Optional[int] = 10

    # multiregion, random, largest
    greedy_match_multiregion: bool = False
    # ignore, balanced_match
    non_matched_region_mode: str = 'balanced_match'
    cost_coeff_cls: float = 1.0
    cost_coeff_bbox: float = 5.0
    cost_coeff_giou: float = 2.0
    cost_coeff_weights: float = 3.0
    loss_coeff_bbox: float = 5.0
    loss_coeff_giou: float = 2.0
    loss_coeff_weights: float = 3.0

    weights_loss_auto_weight: bool = True

    # --- Loss: Pathology Classification (patho-cls) ---
    use_patho_cls: bool = False
    coeff_patho_cls: float = 0.0
    subsample_patho_cls: Optional[int] = 10
    patho_cls_temp: float =  0.2 
    use_negatives_of_other_classes: bool = True
    ignore_other_classes_when_false_target: bool = False


class PathologyTokenSupervisor(nn.Module):
    def __init__(self, config: PathologyTokenConfig, main_config):
        super().__init__()
        self.config: PathologyTokenConfig = prepare_config(config, PathologyTokenConfig, log)

        self.requires_sentence_tokens = False
        self.requires_anatomy_tokens = False
        self.requires_pathology_tokens = True
        self.requires_region_pathology_tokens = False

        assert self.config.non_matched_region_mode in ['ignore', 'balanced_match']

        self.box_weights_loss = get_focal_loss(auto_weight=self.config.weights_loss_auto_weight, logits=False)
        
    def forward(self, 
                model: 'ChEX',
                encoded_image: ImageEncoderOutput, 
                patho_pos_prompt_emb: Optional[torch.FloatTensor] = None,
                patho_neg_prompt_emb: Optional[torch.FloatTensor] = None,
                has_class_bboxes: Optional[torch.BoolTensor] = None,
                target_cls_boxes_padded: Optional[torch.FloatTensor] = None,
                target_cls_boxes_mask: Optional[torch.BoolTensor] = None,
                has_class_labels: Optional[torch.BoolTensor] = None,
                target_cls_labels: Optional[torch.FloatTensor] = None,
                **kwargs) -> torch.FloatTensor:
        """
        :param encoded_image
        :param has_class_bboxes: (N x C)
        :param target_cls_boxes: List of length N with tensors of shape (M_i x 5) in (x_c, y_c, w, h, c) format
        :param has_class_labels: (N x C)
        :param target_cls_labels: (N x C)
        """
        assert patho_pos_prompt_emb is not None, 'PathologyTokenSupervisor requires pathology prompts, but the dataset does not provide class_names'
        N = encoded_image.N
        C, d = patho_pos_prompt_emb.shape
        has_class_labels = has_class_labels if has_class_labels is not None else \
            (torch.ones((N, C), dtype=torch.bool, device=encoded_image.device) if target_cls_labels is not None else torch.zeros((N, C), dtype=torch.bool, device=encoded_image.device))

        # ========== Select the relevant samples and classes to consider ==========
        # (C)
        classes_with_boxes = subsample_classes(has_class_bboxes, self.config.subsample_patho_boxes, self.config.use_patho_detect, C=C, device=encoded_image.device)
        classes_with_labels = subsample_classes(has_class_labels, self.config.subsample_patho_cls, self.config.use_patho_cls, C=C, device=encoded_image.device)
        relevant_classes = classes_with_labels | classes_with_boxes
        # (N)
        samples_with_boxes = (classes_with_boxes[None, :] & has_class_bboxes).any(1)
        samples_with_labels = (classes_with_labels[None, :] & has_class_labels).any(1)
        relevant_samples = samples_with_labels | samples_with_boxes
        # (N')
        encoded_image = encoded_image[relevant_samples]
        # (C' x d)
        query_prompt_emb = patho_pos_prompt_emb[relevant_classes]

        # ========== Detect classes in the image (i.e. predict bounding boxes for class queries) ==========
        detected_regions: TokenDetectorOutput = \
            model.detect_prompts(
                encoded_image,
                box_prompts_emb=query_prompt_emb, # (C x d)
                box_prompt_mask=None)

        # ========== Losses ==========
        loss = 0.
        sub_losses = {}

        # --- Loss: Pathology bounding box (patho-det) ---
        if self.config.use_patho_detect:
            classes_with_boxes_in_relevant = classes_with_boxes[relevant_classes]
            samples_with_boxes_in_relevant = samples_with_boxes[relevant_samples]
            bbox_detected_regions = detected_regions[samples_with_boxes_in_relevant][:, classes_with_boxes_in_relevant]

            # note: target_cls_boxes_padded / target_cls_boxes_mask contain only boxes for classes that have boxes (original_classes_with_boxes)
            original_classes_with_boxes = has_class_bboxes.any(0)
            target_cls_boxes_padded = target_cls_boxes_padded[samples_with_boxes][:, classes_with_boxes[original_classes_with_boxes]]
            target_cls_boxes_mask = target_cls_boxes_mask[samples_with_boxes][:, classes_with_boxes[original_classes_with_boxes]]

            has_class_bboxes = has_class_bboxes[samples_with_boxes][:, classes_with_boxes]
            bbox_pos_prompt_emb = patho_pos_prompt_emb[classes_with_boxes]
            bbox_neg_prompt_emb = patho_neg_prompt_emb[classes_with_boxes]

            cls_detect_loss, cls_detect_sub_losses = self.train_patho_detect(
                bbox_detected_regions, bbox_pos_prompt_emb, bbox_neg_prompt_emb,
                target_cls_boxes_padded, target_cls_boxes_mask,
                has_class_bboxes=has_class_bboxes)
            loss += cls_detect_loss
            sub_losses.update(cls_detect_sub_losses)

        # --- Loss: Pathology Classification (patho-cls) ---
        if self.config.use_patho_cls:
            classes_with_labels_in_relevant = classes_with_labels[relevant_classes]
            samples_with_labels_in_relevant = samples_with_labels[relevant_samples]
            clslabel_detected_regions = detected_regions[samples_with_labels_in_relevant][:, classes_with_labels_in_relevant]
           
            has_class_labels = has_class_labels[samples_with_labels][:, classes_with_labels]
            target_cls_labels = target_cls_labels[samples_with_labels][:, classes_with_labels]
            label_pos_prompt_emb = patho_pos_prompt_emb[classes_with_labels]
            label_neg_prompt_emb = patho_neg_prompt_emb[classes_with_labels]

            cls_labels_loss, cls_labels_sub_losses = self.train_patho_cls(
                clslabel_detected_regions, 
                label_pos_prompt_emb, label_neg_prompt_emb,
                target_cls_labels, has_class_labels=has_class_labels)
            loss += cls_labels_loss
            sub_losses.update(cls_labels_sub_losses)

        return loss, sub_losses, {}

    def train_patho_detect(self, 
                detected_regions: TokenDetectorOutput, 
                pos_prompt_emb: torch.Tensor, neg_prompt_emb: torch.Tensor,
                target_cls_boxes_padded: torch.FloatTensor,
                target_cls_boxes_mask: torch.BoolTensor,
                has_class_bboxes: Optional[torch.BoolTensor] = None,) -> torch.FloatTensor:
        """
        :param target_cls_boxes_padded: (N x C x R x 4)
        :param target_cls_boxes_mask: (N x C x R)
        :param has_class_bboxes: (N x C)
        """
        # --- match bounding boxes and targets ---
        target_cls = target_cls_boxes_mask.any(-1) # (N x C)
        # (N x C x R_s x 4)
        pred_boxes = detected_regions.multiboxes
        # (N x C x R_s)
        pred_box_weights = detected_regions.multiboxes_weights
        # (N x C x R_s x d)
        pred_box_features = detected_regions.multiboxes_features
        N, C, R_s, _ = pred_boxes.shape

        # ---> classify regions
        # (1 x C x 1 x d)
        pos_prompt_emb = pos_prompt_emb[None, :, None, :]
        neg_prompt_emb = neg_prompt_emb[None, :, None, :]
        # (N x C x R_s)
        cls_logits, cls_probs, _ = classify_features(
            pred_box_features, pos_prompt_emb, neg_prompt_emb, 
            softmax=True, normalized=False, 
            return_logits=True)

        # ---> compute costs
        # (N x C x R_s x R_t)
        bbox_cost = bbox_l1_ccost(pred_boxes, target_cls_boxes_padded)
        giou_cost = bbox_giou_ccost(pred_boxes, target_cls_boxes_padded)
        # (N x C x R_s)
        weights_cost = 1. - pred_box_weights
        # (N x C x R_s)
        cls_cost = 1. - cls_probs
        # (N x C x R_s x R_t)
        cost = self.config.cost_coeff_bbox * bbox_cost + \
                + self.config.cost_coeff_giou * giou_cost \
                + self.config.cost_coeff_cls * cls_cost[..., None] \
                + self.config.cost_coeff_weights * weights_cost[..., None]
        
        # ---> match multiregions
        # (N x C x R_s), (N x C x R_s x R_t)
        matches, assign_mask = match_multiregions(cost, mask=target_cls_boxes_mask, non_matched_region_mode=self.config.non_matched_region_mode, greedy_match=self.config.greedy_match_multiregion)

        # ---> compute losses
        # (N x C x R_s x R_t)
        box_loss = self.config.loss_coeff_bbox * bbox_cost + self.config.loss_coeff_giou * giou_cost
        # (N x C)
        # only compute box loss for assigned regions (all matches, not just best matched)
        box_loss = (box_loss * assign_mask).sum(dim=-1).sum(dim=-1) / assign_mask.sum(dim=-1).sum(dim=-1).clamp(min=1)
        # only compute box loss for positive classes
        box_loss = target_cls * box_loss
        box_loss = (has_class_bboxes * box_loss).sum(dim=-1) / has_class_bboxes.sum(dim=-1).clamp(min=1)
        box_loss = box_loss.mean(dim=-1)

        # weights loss using bce-like loss and matches
        # (N x C)
        weights_loss = self.box_weights_loss(pred_box_weights, matches).mean(dim=-1)
        weights_loss = (has_class_bboxes * weights_loss).sum(dim=-1) / has_class_bboxes.sum(dim=-1).clamp(min=1)
        weights_loss = weights_loss.mean(dim=-1)

        # cls loss using bce-like loss
        # (N x C)
        loss = box_loss + self.config.loss_coeff_weights * weights_loss
        sub_losses = {
            'l_cls/box_loss': box_loss,
            'l_cls/weights_loss': weights_loss,
            'l_cls/patho_detect': loss
        }
        loss = self.config.coeff_patho_detect * loss
        
        return loss, sub_losses
        
    def train_patho_cls(self, 
                detected_regions: TokenDetectorOutput, 
                pos_prompt_emb: torch.Tensor, neg_prompt_emb: torch.Tensor, 
                target_cls_labels: BoolTensor,
                has_class_labels: Optional[torch.BoolTensor] = None,) -> torch.FloatTensor:
        # (N x C x d)
        cls_features = detected_regions.box_features

        loss = 0.0
        sub_losses = {}

        # (N x C)
        patho_cls_loss = pathology_contrastive_loss(cls_features, pos_prompt_emb, neg_prompt_emb, target_cls_labels,
                                                                    temp=self.config.patho_cls_temp,
                                                                    use_negatives_of_other_classes=self.config.use_negatives_of_other_classes,
                                                                    ignore_other_classes_when_false_target=self.config.ignore_other_classes_when_false_target)
        # (N)
        patho_cls_loss = (has_class_labels * patho_cls_loss).sum(dim=-1) / has_class_labels.sum(dim=-1).clamp(min=1)
        patho_cls_loss = patho_cls_loss.mean(dim=0)
        sub_losses['l_cls/patho_cls'] = patho_cls_loss
        loss += self.config.coeff_patho_cls * patho_cls_loss

        return loss, sub_losses
