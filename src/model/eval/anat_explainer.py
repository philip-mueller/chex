from dataclasses import dataclass, field
import dataclasses
from typing import Dict, List, Optional
import einops
from omegaconf import MISSING
from torch import BoolTensor, FloatTensor, Tensor
import torch
from metrics.bootstrapping import BootstrapMetricsWrapper
from metrics.classification_metrics import RegionMultiLabelClassificationMetrics
from metrics.textgen_metrics import SentenceMetrics

from model.detector.token_decoder_detector import TokenDetectorOutput
from util.plot_utils import wandb_plot_text
from util.train_utils import EvalConfig, Evaluator


@dataclass
class AnatomyExplainerOutput:
    # (N x A)
    generated_sentences: Optional[List[List[str]]] = None
    # (N x A)
    target_sentences: Optional[List[List[List[str]]]] = None
    target_anat_present_masks: Optional[BoolTensor] = None

    # (N x A x C)
    region_cls_probs:Optional[Tensor] = None
    # (N x A x C)
    region_cls_preds: Optional[Tensor] = None
    # (N x A x C)
    target_anat_cls_labels: Optional[Tensor] = None


@dataclass
class AnatomyExplainerConfig(EvalConfig):
    anatomy_prompts: Dict[str, List[str]] = field(default_factory=dict)
    
    classify_boxes: bool = True
    pos_prompts: Dict[str, List[str]] = field(default_factory=dict)
    neg_prompts: Dict[str, List[str]] = field(default_factory=dict)
    no_finding_prompts: List[str] = field(default_factory=list)
    neg_prompt_mode: str = 'neg'  # neg, pos_centroid, no_finding
    prompt_threshold: Optional[float] = 0.5
    normalize_classification: bool = True

    generate_sentences: bool = True
    sentence_generartion_kwargs: Dict = field(default_factory=dict)

    # If true: region features are extracted based on the target boxes
    # If false: region are detected based on prompts
    use_target_boxes: bool = True
    skip_roi_pool_inference: bool = True

class AnatomyExplainerEvaluator(Evaluator):
    def __init__(self, config: AnatomyExplainerConfig, model: 'ChEX', bootstrap=False, results_path: Optional[str] = None,  **kwargs):
        super().__init__(config, config_cls=AnatomyExplainerConfig, **kwargs)
        from src.model.chex import ChEX
        self.model: ChEX = model

        assert self.dataset.anatomy_names is not None and len(self.dataset.anatomy_names) > 0, 'Dataset does not have anatomy names (missing anatomy_names in the config)'
        self.anatomy_names = self.dataset.anatomy_names
        if self.config.classify_boxes:
            assert self.dataset.has_anatomy_class_labels
            assert self.dataset.class_names is not None and len(self.dataset.class_names) > 0, 'Dataset does not have class names (missing class_names in the config)'
            self.class_names = self.dataset.class_names
            self.classification_metrics = RegionMultiLabelClassificationMetrics(self.anatomy_names, self.class_names, return_class_metrics=True)
            if bootstrap:
                self.classification_metrics = BootstrapMetricsWrapper(
                                                self.classification_metrics, 
                                                n_bootstrap=250,
                                                csv_path=f'{results_path}_bootstrap.csv' if results_path is not None else None)

             # (C x d)
            self.pos_prompt_emb, self.neg_prompt_emb = self.model.encode_pos_neg_prompts(
                self.config.neg_prompt_mode, self.class_names,
                self.config.pos_prompts, self.config.neg_prompts, self.config.no_finding_prompts)

        if self.config.generate_sentences:
            assert self.dataset.has_anatomy_sentences
            self.sentence_metrics = SentenceMetrics(
                use_meteor=True, use_ce=True, use_ratio=True,
                use_bleu=False, use_rouge=False,use_cider=False, use_ratio=False, 
                micro=True, normal_abnormal=True, region_names=self.anatomy_names,
                ce_per_class=True)
            if bootstrap:
                self.sentence_metrics = BootstrapMetricsWrapper(
                                                    self.sentence_metrics, 
                                                    n_bootstrap=10,
                                                    csv_path=f'{results_path}_sent_bootstrap.csv' if results_path is not None else None)

        if not self.config.use_target_boxes:
            # Encode prompts
            # (A x d) 
            self.anatomy_token_emb, _ = self.model.encode_prompts(self.config.anatomy_prompts, self.anatomy_names)

    def eval_step(self, 
        x: Tensor,
        target_anat_boxes: Optional[FloatTensor] = None, 
        target_anat_present_masks: Optional[BoolTensor] = None,
        target_anat_sentences: Optional[List[List[List[str]]]] = None,
        target_anat_cls_labels: Optional[Tensor] = None,
        **kwargs) -> AnatomyExplainerOutput:

        # 1. Encode region features (using tokens or target anatomy boxes)
        if self.config.use_target_boxes:
            assert target_anat_boxes is not None
            assert target_anat_present_masks is not None
            # (N x A x d)
            region_features = self.model.encode_regions(x, target_anat_boxes, target_anat_present_masks)
        else:
            # Detect anatomical regions
            detected_regions: TokenDetectorOutput = self.model.detect_prompts(x, self.anatomy_token_emb, skip_roi_pool=self.config.skip_roi_pool_inference)
            # (N x A x d)
            region_features = detected_regions.box_features

        # 2. Generate sentences for regions
        if self.config.generate_sentences:
            target_anat_present_masks_cpu = target_anat_present_masks.cpu()
            predicted_anat_sentences: List[List[str]] = self.model.generate_sentences(region_features, target_anat_present_masks, **self.config.sentence_generartion_kwargs)
            # concatenate several sentences for each anatomy
            target_anat_sentences: List[List[str]] = [[' '.join(anat_s) for anat_s in sample_s] for sample_s in target_anat_sentences]
            # only keep sentences for present anatomies
            target_anat_sentences = [[anat_s for anat_s, present in zip(sample_s, present_mask) if present] for sample_s, present_mask in zip(target_anat_sentences, target_anat_present_masks_cpu.numpy())]
            # (N x A)
            anat_ids = einops.repeat(torch.arange(len(self.anatomy_names), device='cpu'), 'a -> n a', n=x.shape[0])
            anat_ids: List[torch.Tensor] = [anat_ids_i[present_mask_i] for anat_ids_i, present_mask_i in zip(anat_ids, target_anat_present_masks_cpu)] 
            is_normal = ~(target_anat_cls_labels.bool().any(dim=-1))
            is_normal = [is_normal_i[present_mask_i] for is_normal_i, present_mask_i in zip(is_normal, target_anat_present_masks_cpu)]
            self.sentence_metrics.update(preds=predicted_anat_sentences, targets=target_anat_sentences, region_ids=anat_ids, is_normal=is_normal)
        else:
            predicted_anat_sentences = None

        # 3. classify regions
        if self.config.classify_boxes:
            assert target_anat_cls_labels is not None
            N, A, d = region_features.shape
        
            # Classify
            C, d = self.pos_prompt_emb.shape
            # (N x A x C x d)
            repeated_region_features = einops.repeat(region_features, 'n a d -> n a c d', c=C)
            # (N x A x C x d)
            pos_prompt_emb = einops.repeat(self.pos_prompt_emb, 'c d -> n a c d', n=N, a=A)
            neg_prompt_emb = einops.repeat(self.neg_prompt_emb, 'c d -> n a c d', n=N, a=A)
        
            # (N x A x C)
            region_cls_probs, region_cls_preds = self.model.classify_features(
                repeated_region_features, pos_prompt_emb, neg_prompt_emb, 
                threshold=self.config.prompt_threshold, normalized=self.config.normalize_classification)
            
            self.classification_metrics.update(anatomy_observation_preds=region_cls_preds, anatomy_observation_probs=region_cls_probs, target_anatomy_observations=target_anat_cls_labels, anatomy_masks=target_anat_present_masks)
        else:
            region_cls_probs = None
            region_cls_preds = None

        output = AnatomyExplainerOutput(
            generated_sentences=predicted_anat_sentences,
            target_sentences=target_anat_sentences,
            target_anat_present_masks=target_anat_present_masks,
            region_cls_probs=region_cls_probs,
            region_cls_preds=region_cls_preds,
            target_anat_cls_labels=target_anat_cls_labels)

        return output
    
    def _compute_metrics(self) -> dict:
        metrics = {}
        if self.config.classify_boxes:
            metrics.update(self.classification_metrics.compute())
        if self.config.generate_sentences:
            metrics.update(self.sentence_metrics.compute())
        return metrics
    
    def plot(self, output: AnatomyExplainerOutput, max_samples: int, target_dir: str, plot_local):
        plots = {}
        if self.config.generate_sentences:
            plots['gen_tex_anatt'] = wandb_plot_text(output.generated_sentences, output.target_sentences, max_samples=max_samples)
            
        return plots
