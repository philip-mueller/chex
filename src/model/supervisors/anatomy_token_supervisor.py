from functools import partial
import logging
from typing import Dict, List, Optional
import einops
from model.components.contrastive_losses import anat_pathology_contrastive_loss, sentence_mse_loss
import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass, field
from omegaconf import MISSING
from model.components.bbox_losses import bbox_giou_ccost, bbox_giou_pcost, bbox_l1_ccost, bbox_l1_pcost, match_multiregions
from model.detector.token_decoder_detector import TokenDetectorOutput

from model.img_encoder import ImageEncoderOutput
from model.supervisors.utils import subsample_anat_classes, subsample_anat_regions
from model.txt_encoder import TextEncoderOutput
from util.model_utils import prepare_config

log = logging.getLogger(__name__)

@dataclass
class AnatomyTokenConfig:
    # --- General ---
    # token, bbox, random
    region_encoding_mode: str = 'token'

    # --- Loss: Anatomy bounding box (anat-det) ---
    use_anat_det: bool = False
    coeff_anat_det: float = 0.0
    subsample_anat_det: Optional[int] = 20

    loss_coeff_bbox: float = 5.
    loss_coeff_giou: float = 2.
    cost_coeff_bbox: float = 5.
    cost_coeff_giou: float = 2.
    cost_coeff_weights: float = 3.

    multiregions_match_all_boxes: bool = True
    greedy_match_multiregion: bool = False

    # --- Loss: Anatomy Classification (anat-cls) ---
    use_anat_cls: bool = False
    coeff_anat_cls: float = 0.0
    subsample_anat_cls: Optional[int] = None
    subsample_anat_cls_pathologies: Optional[int] = None
    
    anat_cls_temp: float = 0.25
    # (pos, region_pos, neg, region_neg)
    anat_cls_pos: List[str] = field(default_factory=lambda: ['pos', 'neg'])
    # (pos, region_pos, other_region_pos, neg, region_neg, other_region_neg)
    anat_cls_neg: List[str] = field(default_factory=lambda: ['pos', 'neg'])
    anat_cls_subsample_negatives: Optional[int] = 10

    # --- Loss: Anatomy MSE (anat-mse) ---
    use_anat_mse: bool = False
    coeff_anat_mse: float = 0.0

    # --- Loss: Anatomy Sentence Generation (anat-gen) ---
    use_anat_gen: bool = False
    coeff_anat_gen: float = 0.0
    # concat, sample
    multi_sentence_mode: str = 'concat'
    # ignore, empty_string, no_finding
    empty_sentences_mode: str = 'ignore'


class AnatomyTokenSupervisor(nn.Module):
    def __init__(self, config: AnatomyTokenConfig, main_config):
        super().__init__()
        self.config: AnatomyTokenConfig = prepare_config(config, AnatomyTokenConfig, log)

        assert self.config.multi_sentence_mode in ['concat', 'sample']
        assert self.config.empty_sentences_mode in ['ignore', 'empty_string', 'no_finding']

        self.requires_sentence_tokens = False
        self.requires_anatomy_tokens = True
        self.requires_pathology_tokens = self.config.use_anat_cls
        self.requires_region_pathology_tokens = self.config.use_anat_cls and \
            any(reg in ['region_pos', 'region_neg', 'other_region_pos', 'other_region_neg'] for reg in self.config.anat_cls_pos + self.config.anat_cls_neg)

    def forward(self, 
                 model: 'ChEX',
                encoded_image: ImageEncoderOutput, 
                anatomy_token_emb: Optional[torch.FloatTensor] = None,
                patho_pos_prompt_emb: Optional[torch.FloatTensor] = None,
                patho_neg_prompt_emb: Optional[torch.FloatTensor] = None,
                region_pos_prompt_emb: Optional[torch.FloatTensor] = None,
                region_neg_prompt_emb: Optional[torch.FloatTensor] = None,
                target_anat_boxes: Optional[torch.FloatTensor] = None,
                target_anat_present_masks: Optional[torch.BoolTensor] = None,
                target_anat_multiboxes: Optional[torch.FloatTensor] = None,
                target_anat_multiboxes_masks: Optional[torch.BoolTensor]= None,
                has_anatomy_bboxes: Optional[torch.BoolTensor] = None,
                has_anatomy_multiboxes: Optional[torch.BoolTensor] = None,
                target_anat_sentences: Optional[List[List[List[str]]]] = None,
                has_anatomy_sentences: Optional[torch.BoolTensor] = None,
                target_anat_cls_labels: Optional[torch.FloatTensor] = None,
                has_anatomy_class_labels: Optional[torch.BoolTensor] = None,
                epoch=None,
                **kwargs) -> torch.FloatTensor:
        """
        :param encoded_image
        ---
        :param anatomy_token_emb: (A x d)
        :param patho_pos_prompt_emb: (C x d)
        :param patho_neg_prompt_emb: (C x d)
        :param region_pos_prompt_emb: (A x C x d)
        :param region_neg_prompt_emb: (A x C x d)
        ---
        :param target_anat_boxes: (N x A_single x 4)
        :param target_anat_present_masks: (N x A_single)
        :param has_anatomy_bboxes: (N x A_single)
        :param target_anat_multiboxes_padded: (N x A_multi x R x 4)
        :param target_anat_multiboxes_mask: (N x A_multi x R)
        :param has_anatomy_multiboxes: (N x A_multi)
        ---
        :param target_anat_sentences: List[List[List[str]]]
        :param has_anatomy_sentences: (N x A)
        :param target_anat_cls_labels: (N x A x C)
        :param has_anatomy_class_labels: (N x A x C)
        """
        assert anatomy_token_emb is not None, 'AnatomyTokenSupervisor requires anatomy prompts, but the dataset does not provide anatomy_names'
        N = encoded_image.N
        A_all, d = anatomy_token_emb.shape    
        
        assert self.config.region_encoding_mode in ['token', 'bbox', 'random']
        if self.config.region_encoding_mode == 'token':
            encode_tokens = True
        elif self.config.region_encoding_mode == 'bbox':
            assert not self.config.use_anat_det
            encode_tokens = False
        elif self.config.region_encoding_mode == 'random':
            assert not self.config.use_anat_det
            encode_tokens = np.random.rand() < 0.5
        else:
            raise ValueError(f'Unknown region_encoding_mode: {self.config.region_encoding_mode}')   
    
         # ========== Select the relevant samples and classes to consider ==========
        relevant_anat_regions = torch.zeros((A_all,), dtype=torch.bool, device=encoded_image.device)
        relevant_samples = torch.zeros((N,), dtype=torch.bool, device=encoded_image.device)
        if self.config.use_anat_det: #  or self.config.reconstruct_mse or self.config.train_generation:
            assert target_anat_boxes is not None
            assert target_anat_present_masks is not None
        if target_anat_boxes is not None:
            A_single = target_anat_boxes.shape[1]
            has_anatomy_bboxes = has_anatomy_bboxes if has_anatomy_bboxes is not None else torch.ones((N, A_single), dtype=torch.bool, device=encoded_image.device)
            anat_regions_with_single_bboxes = subsample_anat_regions(has_anatomy_bboxes, self.config.subsample_anat_det, True, A=A_single, device=encoded_image.device)

            if target_anat_multiboxes is not None and encode_tokens:
                assert target_anat_multiboxes_masks is not None
                A_multi = target_anat_multiboxes.shape[1]
                has_anatomy_multiboxes = has_anatomy_multiboxes if has_anatomy_multiboxes is not None else torch.ones((N, A_multi), dtype=torch.bool, device=encoded_image.device)
            else:
                A_multi = 0
                A_all = A_single
                has_anatomy_multiboxes = torch.zeros((N, 0), dtype=torch.bool, device=encoded_image.device)
            anat_regions_with_multi_bboxes = has_anatomy_multiboxes.any(0)
            # (A)
            anat_regions_with_bboxes = torch.cat([anat_regions_with_single_bboxes, anat_regions_with_multi_bboxes])
            has_region_bboxes = torch.cat([has_anatomy_bboxes, has_anatomy_multiboxes], dim=1)
            # (N)
            samples_with_boxes = (anat_regions_with_bboxes[None, :] & has_region_bboxes).any(1)

            relevant_anat_regions = anat_regions_with_bboxes
            relevant_samples = samples_with_boxes
    
        if self.config.use_anat_cls:
            assert target_anat_cls_labels is not None
            assert patho_pos_prompt_emb is not None
            A_single, C = target_anat_cls_labels.shape[1:]
            A_multi = A_all - A_single
            has_anatomy_class_labels = has_anatomy_class_labels if has_anatomy_class_labels is not None else torch.ones((N, A_single, C), dtype=torch.bool, device=encoded_image.device)

            # (A_single x C)
            anat_cls_with_labels = subsample_anat_classes(has_anatomy_class_labels, 
                                                            self.config.subsample_anat_cls, self.config.subsample_anat_cls_pathologies, 
                                                            self.config.use_anat_cls, 
                                                            C=C, A=A_single, device=encoded_image.device)
            
            anat_single_regions_with_labels = anat_cls_with_labels.any(1)
            if A_multi > 0:
                anat_regions_with_labels = torch.cat([anat_single_regions_with_labels, torch.zeros((A_multi,), dtype=torch.bool, device=encoded_image.device)])
            else:
                anat_regions_with_labels = anat_single_regions_with_labels
            # (C)
            classes_with_labels = anat_cls_with_labels.any(0)
            samples_with_labels = (anat_cls_with_labels[None, :, :] & has_anatomy_class_labels).any(2).any(1)
            relevant_anat_regions = relevant_anat_regions | anat_regions_with_labels
            relevant_samples = relevant_samples | samples_with_labels
        
        if self.config.use_anat_mse or self.config.use_anat_gen:
            assert target_anat_sentences is not None
            # (N x A_single)
            has_anatomy_sentences = has_anatomy_sentences if has_anatomy_sentences is not None else torch.ones((N, A_single), dtype=torch.bool, device=encoded_image.device)
            has_anatomy_sentences = has_anatomy_sentences & anat_regions_with_single_bboxes[None, :] & samples_with_boxes[:, None]
            samples_with_sentences = has_anatomy_sentences.any(1)
            anat_single_regions_with_sentences = has_anatomy_sentences.any(0)
            anat_regions_with_sentences = torch.cat([anat_single_regions_with_sentences, torch.zeros((A_multi,), dtype=torch.bool, device=encoded_image.device)]) if A_multi > 0 else anat_single_regions_with_sentences
        
        # (N')
        encoded_image = encoded_image[relevant_samples]

        # ========== Detect the anatimcal regions ==========
        if encode_tokens:
            anatomy_token_emb = anatomy_token_emb[relevant_anat_regions]
            detected_regions: TokenDetectorOutput = \
                model.detect_prompts(
                    encoded_image,
                    box_prompts_emb=anatomy_token_emb, # (A' x d)
                    box_prompt_mask=None)
            # (N' x A' x d)
            region_features = detected_regions.box_features
        else:
            assert target_anat_boxes is not None
            region_features = model.encode_regions(encoded_image, target_anat_boxes[relevant_samples][:, relevant_anat_regions[:A_single]])
            detected_regions = None
        
       
         # ========== Loss Functions ==========
        loss = 0.0
        sub_losses = {}

        # --- Loss: Anatomy bounding boxx (anat-det) ---
        if self.config.use_anat_det:
            assert target_anat_boxes is not None
            assert target_anat_present_masks is not None
            target_anat_boxes = target_anat_boxes[samples_with_boxes][:, anat_regions_with_single_bboxes]
            has_anatomy_bboxes = has_anatomy_bboxes[samples_with_boxes][:, anat_regions_with_single_bboxes] 
            target_anat_present_masks_for_bbox = target_anat_present_masks[samples_with_boxes][:, anat_regions_with_single_bboxes] & has_anatomy_bboxes
            target_anat_multiboxes = target_anat_multiboxes[samples_with_boxes][:, anat_regions_with_multi_bboxes] if target_anat_multiboxes is not None else None
            has_anatomy_multiboxes = has_anatomy_multiboxes[samples_with_boxes][:, anat_regions_with_multi_bboxes] if has_anatomy_multiboxes is not None else None
            target_anat_multiboxes_masks = target_anat_multiboxes_masks[samples_with_boxes][:, anat_regions_with_multi_bboxes] & has_anatomy_multiboxes[:, :, None] if target_anat_multiboxes_masks is not None else None

            anat_with_boxes_in_relevant = anat_regions_with_bboxes[relevant_anat_regions]
            samples_with_boxes_in_relevant = samples_with_boxes[relevant_samples]
            bbox_detected_regions = detected_regions[samples_with_boxes_in_relevant][:, anat_with_boxes_in_relevant, ...]
            anatomy_token_emb = anatomy_token_emb[anat_with_boxes_in_relevant]

            anat_detect_loss, anat_detect_sub_losses = self.train_anat_det(
                                                                            bbox_detected_regions,
                                                                            target_anat_boxes,
                                                                            target_anat_present_masks_for_bbox,
                                                                            target_anat_multiboxes,
                                                                            target_anat_multiboxes_masks)
            loss += anat_detect_loss
            sub_losses.update(anat_detect_sub_losses)

        # --- Loss: Anatomy Classification (anat-cls) ---
        if self.config.use_anat_cls:
            assert target_anat_cls_labels is not None
            anat_regions_with_labels_in_relevant = anat_regions_with_labels[relevant_anat_regions]
            samples_with_labels_in_relevant = samples_with_labels[relevant_samples]
            # (N' x A' x d)
            region_features_for_cls = region_features[samples_with_labels_in_relevant][:, anat_regions_with_labels_in_relevant]
            # (N' x A' x C')
            target_anat_cls_labels = target_anat_cls_labels[samples_with_labels][:, anat_single_regions_with_labels][:, :, classes_with_labels]
            # (N' x A' x C')
            has_anatomy_class_labels = has_anatomy_class_labels[samples_with_labels][:, anat_single_regions_with_labels][:, :, classes_with_labels]
            if target_anat_present_masks is not None:
                target_anat_present_masks_for_cls = target_anat_present_masks[samples_with_labels][:, anat_single_regions_with_labels]
                has_anatomy_class_labels = has_anatomy_class_labels & target_anat_present_masks_for_cls[:, :, None]
            # (C')
            patho_pos_prompt_emb = patho_pos_prompt_emb[classes_with_labels]
            patho_neg_prompt_emb = patho_neg_prompt_emb[classes_with_labels]
            if region_pos_prompt_emb is not None:
                region_pos_prompt_emb = region_pos_prompt_emb[anat_single_regions_with_labels][:, classes_with_labels]
                region_neg_prompt_emb = region_neg_prompt_emb[anat_single_regions_with_labels][:, classes_with_labels]

            anat_cls_loss, anat_cls_sub_losses = self.train_anat_cls(
                                                        region_features_for_cls, 
                                                        patho_pos_prompt_emb, 
                                                        patho_neg_prompt_emb, 
                                                        region_pos_prompt_emb, 
                                                        region_neg_prompt_emb, 
                                                        target_anat_cls_labels, 
                                                        has_anatomy_class_labels)
            loss += anat_cls_loss
            sub_losses.update(anat_cls_sub_losses)

        # --- Loss: Anatomy MSE (anat-mse) and Sentence Generation (anat-gen) ---
        if self.config.use_anat_mse or self.config.use_anat_gen:
            assert target_anat_sentences is not None

            anat_regions_with_sentences_in_relevant = anat_regions_with_sentences[relevant_anat_regions]
            samples_with_sentences_in_relevant = samples_with_sentences[relevant_samples]
            # (N' x A' x d)
            region_features_for_gen = region_features[samples_with_sentences_in_relevant][:, anat_regions_with_sentences_in_relevant]
            
            if target_anat_present_masks is not None:
                has_anatomy_sentences = has_anatomy_sentences & target_anat_present_masks
            has_anatomy_sentences_for_gen = has_anatomy_sentences[samples_with_sentences][:, anat_regions_with_sentences]

            # Loss: Anatomy MSE (anat-mse)
            if self.config.use_anat_mse: 
                encoded_anat_sentences: TextEncoderOutput = model.encode_region_sentences(target_anat_sentences, device=encoded_image.device)
                # (N x A x S x d)
                anat_sentence_features = encoded_anat_sentences.sentence_features
                anat_sentence_mask = encoded_anat_sentences.sentence_mask
                # (N' x A' x S x d)
                anat_sentence_features = anat_sentence_features[samples_with_sentences][:, anat_single_regions_with_sentences]
                anat_sentence_mask = anat_sentence_mask[samples_with_sentences][:, anat_single_regions_with_sentences]

                # (N' x A' x d)
                anat_sentence_features = (anat_sentence_features * anat_sentence_mask[..., None]).sum(dim=-2) / anat_sentence_mask.sum(dim=-1).clamp(min=1)[..., None]
                anat_sentence_mask = anat_sentence_mask.any(dim=-1)

                sub_losses['l_anat/sent_mse'] = sentence_mse_loss(
                    anat_sentence_features, # (N' x A' x d)
                    region_features_for_gen, # (N x A x d)
                    anat_sentence_mask) # (N x A) 
                
                loss += self.config.coeff_anat_mse * sub_losses['l_anat/sent_mse']

            # Loss: Sentence Generation (anat-gen)
            if self.config.use_anat_gen:
                assert self.config.multi_sentence_mode in ['concat', 'sample']
                if self.config.multi_sentence_mode == 'concat':
                    # concatenate the sentences per anatomy
                    target_anat_text: List[List[str]] = [
                        [' '.join(anat_sents) if anat_sents is not None else '' for anat_sents in sample_sents]
                        for sample_sents in target_anat_sentences
                    ] 
                else:
                    # sample one sentence per anatomy
                    target_anat_text: List[List[str]] = [
                        [
                            np.random.choice(anat_sents, replace=False)
                            if anat_sents is not None and len(anat_sents) > 0 else '' for anat_sents in sample_sents
                        ]
                        for sample_sents in target_anat_sentences
                    ]
                if self.config.empty_sentences_mode == 'ignore':
                    # (N x A)
                    empty_sentence_mask = torch.tensor(
                        [[anat_sent.strip() == '' for anat_sent in sample_sents] for sample_sents in target_anat_text],
                        dtype=torch.bool, device=encoded_image.device)
                    has_anatomy_sentences = has_anatomy_sentences & ~empty_sentence_mask
                    has_anatomy_sentences_for_gen = has_anatomy_sentences_for_gen & ~empty_sentence_mask[samples_with_sentences][:, anat_regions_with_sentences]
                elif self.config.empty_sentences_mode == 'no_finding':
                    target_anat_text = [
                        ['No indication of pathologies.' if anat_sent.strip() == '' else anat_sent for anat_sent in sample_sents]
                        for sample_sents in target_anat_text
                    ]
                else:
                    assert self.config.empty_sentences_mode == 'empty_string'
                            
                assert all(len(sample_sents) == A_single for sample_sents in target_anat_sentences), \
                    'Number of anatomy regions in sentences supervision does not match the number of anatomy regions'
                #flattened_sentences = [
                #    anat_text for sample_sents in target_anat_text for anat_text in sample_sents
                #]

                anat_sent_loss = model.train_sentence_decoder(
                    flattened_features=region_features_for_gen[has_anatomy_sentences_for_gen], # (N' x A' x d)
                    sentence_mask=has_anatomy_sentences,  # (N x A)
                    sentences=target_anat_text,  # (N x A)
                    epoch=epoch
                )
                sub_losses['l_anat/anat_gen'] = anat_sent_loss
                loss += self.config.coeff_anat_gen * anat_sent_loss

        return loss, sub_losses, {}
            
    def train_anat_det(self, 
                    detected_regions: TokenDetectorOutput, 
                    target_anat_boxes: torch.Tensor,  # (N x A x 4)
                    target_anat_present_masks: torch.Tensor,  # (N x A)
                    target_anat_multiboxes_padded: Optional[torch.FloatTensor],  # (N x A x R x 4)
                    target_anat_multiboxes_masks: Optional[torch.BoolTensor]): # (N x A x R)
        N, A_single = target_anat_boxes.shape[:2]
        A_multi = target_anat_multiboxes_padded.shape[1] if target_anat_multiboxes_padded is not None else 0
        losses = {}

        # ----- Single Anatomy Boxes (i.e. one box per anatomy region) -----
        if self.config.multiregions_match_all_boxes and detected_regions.multiboxes is not None:
            _, _, R, _ = detected_regions.multiboxes.shape
            region_boxes = einops.rearrange(detected_regions.multiboxes[:, :A_single], 'n a r d -> n (a r) d', r=R)
            target_anat_boxes = einops.repeat(target_anat_boxes, 'n a d -> n (a r) d', r=R)
            target_anat_present_masks_box = einops.repeat(target_anat_present_masks, 'n a -> n (a r)', r=R)
        else:
            region_boxes = detected_regions.boxes[:, :A_single]
            target_anat_present_masks_box = target_anat_present_masks

        loss_single, num_boxes_single = self.anatomy_single_detection_loss(region_boxes, target_anat_boxes, target_anat_present_masks_box)
        loss_single = loss_single.sum()

        # ----- Multi Anatomy Boxes (i.e. multiple boxes per anatomy region) -----
        if A_multi > 0:
            assert detected_regions.multiboxes is not None
            assert detected_regions.multiboxes.shape[1] == A_single + A_multi
            # (N x A_multi x R_s x 4)
            pred_multiboxes = detected_regions.multiboxes[:, A_single:]
            # (N x A x R_s)
            pred_box_weights = detected_regions.multiboxes_weights[:, A_single:]
            
            loss_multi, num_boxes_multi = self.anatomy_multi_detection_loss(pred_multiboxes, pred_box_weights, target_anat_multiboxes_padded, target_anat_multiboxes_masks)
            loss_multi = loss_multi.sum()
        else:
            loss_multi = 0.
            num_boxes_multi = 0

        loss = (loss_single + loss_multi) / (num_boxes_single + num_boxes_multi).clamp(min=1)
        losses['l_anat/anat_single'] = loss_single.sum() / num_boxes_single.clamp(min=1)
        if A_multi > 0:
            losses['l_anat/anat_multi'] = loss_multi.sum() / num_boxes_multi.clamp(min=1)

        losses['l_anat/anat_det'] = loss
        loss = self.config.coeff_anat_det * loss
        return loss, losses

    def anatomy_single_detection_loss(self, preditecd_boxes, target_boxes, target_mask):
        # (N*A x 4)
        #preditecd_boxes = preditecd_boxes.flatten(0, 1)
        #target_boxes = target_boxes.flatten(0, 1)
        #target_mask = target_mask.flatten().to(dtype=preditecd_boxes.dtype)

        num_boxes = target_mask.to(dtype=preditecd_boxes.dtype).sum()
        # (N x A)
        loss_bbox = bbox_l1_pcost(preditecd_boxes, target_boxes)
        # (N x A)
        loss_giou = bbox_giou_pcost(preditecd_boxes, target_boxes)

        loss = self.config.loss_coeff_bbox * loss_bbox + self.config.loss_coeff_giou * loss_giou
        loss = target_mask * loss
  
        return loss, num_boxes
    
    def anatomy_multi_detection_loss(self, pred_multiboxes, pred_box_weights, target_anat_multiboxes_padded, target_anat_multiboxes_mask):
        num_boxes = target_anat_multiboxes_mask.any(dim=-1).to(dtype=pred_multiboxes.dtype).sum()
        # (N x A_multi x R_s x R_t)
        bbox_cost = bbox_l1_ccost(pred_multiboxes, target_anat_multiboxes_padded)
        giou_cost = bbox_giou_ccost(pred_multiboxes, target_anat_multiboxes_padded)
        # (N x A x R_s)
        weights_cost = 1. - pred_box_weights
        # (N x A x R_s x R_t)
        cost = self.config.cost_coeff_bbox * bbox_cost + \
                + self.config.cost_coeff_giou * giou_cost \
                + self.config.cost_coeff_weights * weights_cost[..., None]
        
        # (N x A x R_s), (N x A x R_s x R_t)
        matches, assign_mask = match_multiregions(cost, mask=target_anat_multiboxes_mask, non_matched_region_mode='balanced_match' if self.config.multiregions_match_all_boxes else 'ignore', greedy_match=self.config.greedy_match_multiregion)

        # (N x A)
        # only compute box loss for assigned regions (all matches, not just best matched)
        loss_bbox = (bbox_cost * assign_mask).sum(dim=-1).sum(dim=-1) / assign_mask.sum(dim=-1).sum(dim=-1).clamp(min=1)
        loss_giou = (giou_cost * assign_mask).sum(dim=-1).sum(dim=-1) / assign_mask.sum(dim=-1).sum(dim=-1).clamp(min=1)

        loss = self.config.loss_coeff_bbox * loss_bbox + self.config.loss_coeff_giou * loss_giou
        loss = target_anat_multiboxes_mask.any(dim=-1) * loss

        return loss, num_boxes

    def train_anat_cls(self, region_features, patho_pos_prompt_emb, patho_neg_prompt_emb, region_pos_prompt_emb, region_neg_prompt_emb, target_anat_cls_labels, has_anatomy_class_labels):
        # (N x A)
        loss = anat_pathology_contrastive_loss(region_features, 
                                        patho_pos_prompt_emb, 
                                        patho_neg_prompt_emb, 
                                        region_pos_prompt_emb=region_pos_prompt_emb,
                                        region_neg_prompt_emb=region_neg_prompt_emb,
                                        targets=target_anat_cls_labels,
                                        has_targets=has_anatomy_class_labels,
                                        temp=self.config.anat_cls_temp, normalized=True, 
                                        positive_prompt_examples=self.config.anat_cls_pos, # : Collection[str] = ('pos', 'region_pos', 'neg', 'region_neg'),
                                        negative_prompt_examples=self.config.anat_cls_neg, # : Collection[str] = ('pos', 'region_pos', 'other_region_pos', 'neg', 'region_neg', 'other_region_neg'),
                                        subsample_negatives=self.config.anat_cls_subsample_negatives
                                   )
        
        loss = loss.sum() / has_anatomy_class_labels.any(-1).sum()

        losses = {'l_anat/anat_cls': loss}
        loss = self.config.coeff_anat_cls * loss
        return loss, losses
