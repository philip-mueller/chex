from dataclasses import dataclass, field
import dataclasses
from typing import Dict, List, Optional
import einops
from omegaconf import MISSING
from torch import BoolTensor, FloatTensor, Tensor
import torch
from metrics.bootstrapping import BootstrapMetricsWrapper
from metrics.classification_metrics import MultiClassClassificationMetrics
from metrics.textgen_metrics import SentenceMetrics

from model.detector.token_decoder_detector import TokenDetectorOutput
from util.plot_utils import wandb_plot_text
from util.train_utils import EvalConfig, Evaluator


@dataclass
class BoxExplainerOutput:
    # (N x M_i)
    generated_sentences: Optional[List[List[str]]] = None
    # (N x M_i x C)
    region_cls_probs: Optional[List[Tensor]] = None
    # (N x M_i)
    region_cls_preds: Optional[List[Tensor]] = None   

    # (N x M_i)
    target_sentences: Optional[List[List[List[str]]]] = None
    # (N x M_i x 4/5)
    target_cls_boxes: Optional[List[torch.FloatTensor]] = None


@dataclass
class BoxExplainerConfig(EvalConfig):
    classify_boxes: bool = True
    pos_prompts: Dict[str, List[str]] = field(default_factory=dict)
    neg_prompts: Dict[str, List[str]] = field(default_factory=dict)
    no_finding_prompts: List[str] = field(default_factory=list)
    neg_prompt_mode: str = 'pos_centroid'  # neg, pos_centroid, no_finding
    normalize_classification: bool = True

    generate_sentences: bool = True
    sentence_generartion_kwargs: Dict = field(default_factory=dict)


class BoxExplainerEvaluator(Evaluator):
    def __init__(self, config: BoxExplainerConfig, model: 'ChEX', bootstrap=False,  results_path: Optional[str] = None, **kwargs):
        super().__init__(config, config_cls=BoxExplainerConfig, **kwargs)
        from model.chex import ChEX
        self.model: ChEX = model

        assert self.dataset.has_class_bboxes
        assert self.dataset.class_names is not None and len(self.dataset.class_names) > 0, 'Dataset does not have class names (missing class_names in the config)'
        self.class_names = self.dataset.class_names

        if self.config.classify_boxes:
            self.classification_metrics = MultiClassClassificationMetrics(self.class_names, return_class_metrics=True)
            if bootstrap:
                self.classification_metrics = BootstrapMetricsWrapper(
                                                self.classification_metrics, 
                                                n_bootstrap=250,
                                                csv_path=f'{results_path}_cls_bootstrap.csv' if results_path is not None else None)

             # (C x d)
            self.pos_prompt_emb, self.neg_prompt_emb = self.model.encode_pos_neg_prompts(
                self.config.neg_prompt_mode, self.class_names,
                self.config.pos_prompts, self.config.neg_prompts, self.config.no_finding_prompts)

        if self.config.generate_sentences:
            assert self.dataset.has_class_box_sentences
            self.sentence_metrics = SentenceMetrics(
                use_meteor=True, use_ce=True,
                use_bleu=False, use_rouge=False,use_cider=False, use_ratio=False, 
                micro=True, region_names=self.class_names)
            if bootstrap:
                self.sentence_metrics = BootstrapMetricsWrapper(
                                                self.sentence_metrics, 
                                                n_bootstrap=10,
                                                csv_path=f'{results_path}_sent_bootstrap.csv' if results_path is not None else None)
            
    def eval_step(self, 
        x: Tensor,
        target_cls_boxes: Optional[List[torch.FloatTensor]] = None, 
        target_cls_box_sentences: Optional[List[List[str]]] = None,
        **kwargs) -> BoxExplainerOutput:

        # 1. Encode region features (using tokens or target anatomy boxes)
        assert target_cls_boxes is not None
        assert target_cls_box_sentences is not None

        assert len(target_cls_boxes) == len(target_cls_box_sentences)
        N = len(target_cls_boxes)
        assert all(boxes_i.shape[0] == len(sentences_i) for boxes_i, sentences_i in zip(target_cls_boxes, target_cls_box_sentences))
        assert all(boxes_i.shape[1] == 5 for boxes_i in target_cls_boxes) if self.config.classify_boxes else all(boxes_i.shape[1] >= 4 for boxes_i in target_cls_boxes)

        # 1. Encode region features
        # Convert to (N x M_max x 4/5) format
        M_is = [boxes_i.shape[0] for boxes_i in target_cls_boxes]
        max_M = max(M_is)
        # (N x M_max)
        target_cls_boxe_mask: BoolTensor = torch.stack([
            torch.cat([torch.ones(boxes_i.shape[0], dtype=torch.bool), torch.zeros(max_M - boxes_i.shape[0], dtype=torch.bool)])
            for boxes_i in target_cls_boxes
        ])
        # (N x M_max x 4/5)
        target_cls_boxes_stacked = torch.stack([
            torch.cat([boxes_i, torch.zeros(max_M - boxes_i.shape[0], boxes_i.shape[1])])
            for boxes_i in target_cls_boxes
        ])

        # (N x M_max x d)
        region_features = self.model.encode_regions(x, target_cls_boxes_stacked[:, :, :4], target_cls_boxe_mask)
       
        # 2. Generate sentences for regions
        if self.config.generate_sentences:
            predicted_box_sentences: List[List[str]] = self.model.generate_sentences(region_features, target_cls_boxe_mask, **self.config.sentence_generartion_kwargs)
            predicted_box_sentences = [sent_i[:M_i] for sent_i, M_i in zip(predicted_box_sentences, M_is)]
            sent_class_ids = [boxes_i[:, 4].long() for boxes_i in target_cls_boxes] if self.config.classify_boxes else None
            self.sentence_metrics.update(preds=predicted_box_sentences, targets=target_cls_box_sentences, region_ids=sent_class_ids)
        else:
            predicted_box_sentences = None

        # 3. classify regions
        if self.config.classify_boxes:
            assert target_cls_boxes_stacked is not None
            
            # Classify
            C, d = self.pos_prompt_emb.shape
            # (N x M_max x C x d)
            repeated_region_features = einops.repeat(region_features, 'n m d -> n m c d', c=C)
            # (N x M_max x C x d)
            pos_prompt_emb = einops.repeat(self.pos_prompt_emb, 'c d -> n m c d', n=N, m=max_M)
            neg_prompt_emb = einops.repeat(self.neg_prompt_emb, 'c d -> n m c d', n=N, m=max_M)
        
            # (N x M_max x C)
            region_cls_probs, _ = self.model.classify_features(
                repeated_region_features, pos_prompt_emb, neg_prompt_emb, normalized=self.config.normalize_classification)
            # (N x M_max)
            region_cls_preds = region_cls_probs.argmax(dim=-1)
            # (N_targets x C)
            pred_cls_probs = region_cls_probs[target_cls_boxe_mask]
            # (N_targets)
            pred_cls_preds = region_cls_preds[target_cls_boxe_mask]
            # (N_targets)
            target_classes = torch.cat([boxes_i[:, 4].long() for boxes_i in target_cls_boxes])

            self.classification_metrics.update(preds=pred_cls_preds, pred_probs=pred_cls_probs, targets=target_classes)
        else:
            region_cls_probs = None
            region_cls_preds = None

        output = BoxExplainerOutput(
            generated_sentences=predicted_box_sentences,
            region_cls_probs=region_cls_probs,
            region_cls_preds=region_cls_preds,
            target_sentences=target_cls_box_sentences,
            target_cls_boxes=target_cls_boxes,
        )
        return output
    
    def _compute_metrics(self) -> dict:
        metrics = {}
        if self.config.classify_boxes:
            metrics.update(self.classification_metrics.compute())
        if self.config.generate_sentences:
            metrics.update(self.sentence_metrics.compute())
        return metrics
    
    def plot(self, output: BoxExplainerOutput, max_samples: int, target_dir: str, plot_local):
        plots = {}
        if self.config.generate_sentences:
            plots['gen_text_patho'] = wandb_plot_text(output.generated_sentences, output.target_sentences, max_samples=max_samples)
            
        return plots
