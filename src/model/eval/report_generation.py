

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import einops
from omegaconf import MISSING

import torch
import torch.nn.functional as F
from metrics.bootstrapping import BootstrapMetricsWrapper
from metrics.textgen_metrics import FullReportMetrics
from model.detector.token_decoder_detector import TokenDetectorOutput
import numpy as np
from util.model_utils import BaseModelOutput

from util.train_utils import EvalConfig, Evaluator
from torch import Tensor


@dataclass
class ReportGenerationConfig(EvalConfig):
    use_anatomy_prompts: bool = True
    anatomy_names: List[str] = field(default_factory=list)
    use_pathology_prompts: bool = True
    pathology_names: List[str] = field(default_factory=list)

    ignore_negative_pathologies: bool = False
    ignore_negative_anat_regions: bool = True
    pos_prompts: Dict[str, List[str]] = field(default_factory=dict)
    neg_prompts: Dict[str, List[str]] = field(default_factory=dict)
    no_finding_prompts: List[str] = field(default_factory=list)
    neg_prompt_mode: str = 'neg'  # neg, pos_centroid, no_finding
    cls_prompt_average: bool = True
    anatomy_prompts: Dict[str, List[str]] = field(default_factory=dict)
    pathology_prompts: Dict[str, List[str]] = field(default_factory=dict)

    positive_threshold: float = 0.5

    synonym_removal_thres: Optional[float] = 0.95
    # remove synonyms if max sentences is reached
    synonyme_soft_removal_thres: Optional[float] = 0.9
   
    # weight, positiveness
    sentence_preference: str = 'weight'

    max_sentences: int = 10

    skip_roi_pool_inference: bool = True

    sentence_generartion_kwargs: Dict = field(default_factory=dict)

@dataclass
class ReportGenerationOutput(BaseModelOutput):
    sample_ids: List[str] = MISSING
    # (N x H x W)
    image: torch.Tensor = MISSING

    pred_sentences: List[List[str]] = MISSING
    pred_boxes: List[List[torch.FloatTensor]] = MISSING
    pred_boxes_weights: List[List[torch.FloatTensor]] = MISSING
    pred_prompt_types: List[List[str]] = MISSING
    pred_prompt_names: List[List[str]] = MISSING  # class names / anatomy names

    target_sentences: List[List[str]] = MISSING

    def split_samples(self):
        N = len(self.sample_ids)
        return [
            ReportGenerationOutput(
                sample_ids=[self.sample_ids[i]],
                image=self.image[None, i],
                pred_sentences=[self.pred_sentences[i]],
                pred_boxes=[self.pred_boxes[i]],
                pred_boxes_weights=[self.pred_boxes_weights[i]],
                pred_prompt_types=[self.pred_prompt_types[i]],
                pred_prompt_names=[self.pred_prompt_names[i]],
                target_sentences=[self.target_sentences[i]],
            )
            for i in range(N)
        ]

class ReportEvaluator(Evaluator):
    def __init__(self, config: ReportGenerationConfig, model: 'ChEX',  bootstrap=False, results_path: Optional[str] = None,  **kwargs):
        super().__init__(config, config_cls=ReportGenerationConfig, **kwargs)
        from src.model.chex import ChEX
        self.model: ChEX = model

        self.pathology_names = self.config.pathology_names
        self.anatomy_names = self.config.anatomy_names
        self.prompt_texts = []
        self.prompt_types = []

        token_prompts = []
        self.n_patho = 0
        if self.config.use_pathology_prompts:
            patho_prompt_emb, _ = self.model.encode_prompts(self.config.pathology_prompts, self.pathology_names)
            token_prompts.append(patho_prompt_emb)
            self.n_patho = len(self.pathology_names)
            self.prompt_texts.extend([self.config.pathology_prompts[name][0] for name in self.pathology_names])
            self.prompt_types.extend(['pathology'] * self.n_patho)
        self.n_anat = 0
        if self.config.use_anatomy_prompts:
            anat_prompt_emb, _ = self.model.encode_prompts(self.config.anatomy_prompts, self.anatomy_names)
            token_prompts.append(anat_prompt_emb)
            self.n_anat = len(self.anatomy_names)
            self.prompt_texts.extend([self.config.anatomy_prompts[name][0] for name in self.anatomy_names])
            self.prompt_types.extend(['anatomy'] * self.n_anat)
        self.token_prompts = torch.cat(token_prompts, dim=0)

        # (C x d)
        self.pos_prompt_emb, self.neg_prompt_emb = self.model.encode_pos_neg_prompts(
            self.config.neg_prompt_mode, self.pathology_names,
            self.config.pos_prompts, self.config.neg_prompts, self.config.no_finding_prompts)
        if self.config.cls_prompt_average:
            self.pos_prompt_emb = self.pos_prompt_emb.mean(dim=0)
            self.neg_prompt_emb = self.neg_prompt_emb.mean(dim=0)

        self.report_metrics = FullReportMetrics(use_bleu=True, use_rouge=True, use_meteor=True, use_cider=True, use_ce=True, ce_per_class=True)
        if bootstrap:
            self.report_metrics = BootstrapMetricsWrapper(
                                                self.report_metrics, 
                                                n_bootstrap=10,
                                                csv_path=f'{results_path}_bootstrap.csv' if results_path is not None else None)
        
    def _predict(self, 
        x: Tensor,
        sentences: List[List[str]],
        sample_id: List[str],
        **kwargs): # -> BoxExplainerOutput:
        
        # 1. Detect prompts
        detected_regions: TokenDetectorOutput = self.model.detect_prompts(
            x, self.token_prompts, clip_boxes=True, skip_roi_pool=self.config.skip_roi_pool_inference)
        # (N x Q x d)
        region_features = detected_regions.box_features
        N, Q, d = region_features.shape
        region_weights = detected_regions.boxes_weights
        
        # (N x Q x R x 4)
        region_multiboxes = detected_regions.multiboxes
        region_multibox_weights = detected_regions.multiboxes_weights

        # 2. Classify regions + select them
        if self.config.cls_prompt_average:
            # (N x Q x d)
            pos_prompt_emb = einops.repeat(self.pos_prompt_emb, 'd -> n q d', n=N, q=Q)
            neg_prompt_emb = einops.repeat(self.neg_prompt_emb, 'd -> n q d', n=N, q=Q)
        
            # (N x Q)
            region_cls_probs, region_cls_preds = self.model.classify_features(
                region_features, pos_prompt_emb, neg_prompt_emb, 
                threshold=self.config.positive_threshold, normalized=True)
        else:
            C, _ = self.pos_prompt_emb.shape
            # (N x Q x d)
            pos_prompt_emb = einops.repeat(self.pos_prompt_emb, 'c d -> n q c d', n=N, q=Q)
            neg_prompt_emb = einops.repeat(self.neg_prompt_emb, 'c d -> n q c d', n=N, q=Q)
            repeated_region_features = einops.repeat(region_features, 'n q d -> n q c d', c=C)
            # (N x Q x C)
            region_cls_probs, region_cls_preds = self.model.classify_features(
                repeated_region_features, pos_prompt_emb, neg_prompt_emb, 
                threshold=self.config.positive_threshold, normalized=True)
            region_cls_probs = region_cls_probs.amax(dim=-1)
            region_cls_preds = region_cls_preds.any(dim=-1)
        
        region_mask = torch.ones_like(region_weights, dtype=torch.bool)
        if self.config.ignore_negative_pathologies and self.config.use_pathology_prompts:
            #region_mask[:, self.n_patho:] = region_cls_preds[:, self.n_patho:]
            region_mask[:, :self.n_patho] = region_cls_preds[:, :self.n_patho]
        if self.config.ignore_negative_anat_regions and self.config.use_anatomy_prompts:
            if self.config.use_pathology_prompts:
                #region_mask[:, :self.n_patho] = region_cls_preds[:, :self.n_patho]
                region_mask[:, self.n_patho:] = region_cls_preds[:, self.n_patho:]
            else:
                region_mask[:, :] = region_cls_preds[:, :]

        # 3. Generate sentences for selected regions
        predicted_box_sentences: List[List[str]] = self.model.generate_sentences(region_features, region_mask, **self.config.sentence_generartion_kwargs)

        return sample_id, x, sentences, predicted_box_sentences, region_mask, region_weights, region_cls_probs, region_multiboxes, region_multibox_weights

    def _postprocess(self, sample_id, x, sentences, predicted_box_sentences, region_mask, region_weights, region_cls_probs, region_multiboxes, region_multibox_weights, config):
        target_sentences = sentences
        # -----> Check for synonyms
        encoded_sentences = self.model.encode_sentences(predicted_box_sentences, device=x.device)
        # (N x Q x d)
        sentence_features = encoded_sentences.sentence_features
        region_mask = encoded_sentences.sentence_mask
        # (N x Q x Q) <- compute pairwise cosine similarity
        sentence_features = F.normalize(sentence_features, dim=-1)
        sentence_similarity = sentence_features @ sentence_features.transpose(1, 2)
        # renormalize to [0, 1] and set diagonal to 0
        sentence_similarity = (sentence_similarity + 1) / 2
        N, Q, _ = sentence_similarity.shape
        sentence_similarity[~region_mask[:, :, None].expand(N, Q, Q)] = 0.
        sentence_similarity[~region_mask[:, None, :].expand(N, Q, Q)] = 0.
        sentence_similarity.diagonal(dim1=1, dim2=2).fill_(0.)

        if config.sentence_preference == 'weight':
            sentence_preference = region_weights
        elif config.sentence_preference == 'positiveness':
            sentence_preference = region_cls_probs
        else:
            raise ValueError(f'Unknown sentence preference {config.sentence_preference}')

        region_masks = [
            self.remove_synonyms(sent_sim, sent_pref, mask, config=config)
            for sent_sim, sent_pref, mask
            in zip(sentence_similarity.cpu(), sentence_preference.cpu(), region_mask.cpu())
        ]

        # -----> Select final sentences and optionally reorder them
        sentences_indices = []
        for sentences, mask in zip(predicted_box_sentences, region_masks):
            sentences_indices.append([i for i, m in enumerate(mask) if m and i < len(sentences)])
        pred_sentences = [
            [sentences[i] for i in indices]
            for sentences, indices in zip(predicted_box_sentences,sentences_indices)
        ]

        pred_boxes = [
            [boxes[i, :] for i in indices]
            for boxes, indices in zip(region_multiboxes, sentences_indices)
        ]
        pred_boxes_weights = [
            [weights[i] for i in indices]
            for weights, indices in zip(region_multibox_weights, sentences_indices)
        ]
        pred_prompt_types = [
            [self.prompt_types[i] for i in indices]
            for indices in sentences_indices
        ]
        pred_prompt_names = [
            [self.prompt_texts[i] for i in indices]
            for indices in sentences_indices
        ]

        return ReportGenerationOutput(
            sample_ids=sample_id,
            image=x,
            pred_sentences=pred_sentences,
            pred_boxes=pred_boxes,
            pred_boxes_weights=pred_boxes_weights,
            pred_prompt_types=pred_prompt_types,
            pred_prompt_names=pred_prompt_names,
            target_sentences=target_sentences,
        )

    def _update_metrics_with_output(self, output: ReportGenerationOutput):
        pred_reports = [' '.join(sentences) for sentences in output.pred_sentences]
        target_reports = [' '.join(sentences) for sentences in output.target_sentences]
        self.report_metrics.update(preds=pred_reports, targets=target_reports)
    
    def _compute_metrics(self) -> dict:
        metrics = self.report_metrics.compute()
        return metrics
    
    def compute_metrics_and_ce_labels(self):
        metrics, ce_labels = self.report_metrics.compute(return_ce_labels=True)
        return metrics, ce_labels

    def remove_synonyms(self, sentence_similarity, preference, mask, config):
        """
        Remove synonyms from sentences
        :param sentence_similarity: (Q x Q)
        :param preference: (Q)
        :param mask: (Q)
        """
        # (Q)
        keep_mask = mask.clone()


        while True:
            # Find indices q and q' of most synonymous sentence pair (which has not yet been removed)
            sentence_similarity = sentence_similarity * keep_mask[:, None] * keep_mask[None, :]
            max_sim, max_sim_idx = sentence_similarity.max(dim=1)
            max_sim, q = max_sim.max(dim=0)
            q_prime = max_sim_idx[q]
            n_sent = keep_mask.sum()

            # Stop if max similarity is below threshold or max sentences is reached
            if max_sim < config.synonym_removal_thres and (max_sim < config.synonyme_soft_removal_thres or n_sent <= config.max_sentences):
                break

            # Remove the sentence with the lower preference
            pref_q, pref_q_prime = preference[q], preference[q_prime]
            q_remove = q if pref_q < pref_q_prime else q_prime
            keep_mask[q_remove] = False

        return keep_mask       

