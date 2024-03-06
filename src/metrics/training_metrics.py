
import einops
from torch import nn
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from metrics.detection_metrics import BoxStatistics, batched_box_iou
from metrics.textgen_metrics import SentenceMetrics
from util.data_utils import to_device

from util.train_utils import AvgDictMeter

import logging
log = logging.getLogger(__name__)

class TrainingMetrics(nn.Module):
    def __init__(self, device='cpu', eval_generation=False):
        super().__init__()
        self.step_metrics_meter = AvgDictMeter()
        self.box_stats = BoxStatistics()
        self.register_buffer('count', torch.tensor(0.0))
        self.register_buffer('mean_l2', torch.tensor(0.0))
        self.register_buffer('mean_cos', torch.tensor(0.0))
        self.register_buffer('mean_rank', torch.tensor(0.0))
        self.register_buffer('box_iou', torch.tensor(0.0))

        self.eval_generation = eval_generation
        if eval_generation:
            self.sentence_metrics = SentenceMetrics(micro=True, sample_macro=False, use_ce=False)
            self._has_sentence_updates = False

        self.device = device
        self.to(device)

    def reset(self):
        self.step_metrics_meter.reset()
        self.box_stats.reset()
        self.count.zero_()
        self.mean_l2.zero_()
        self.mean_cos.zero_()
        self.mean_rank.zero_()
        self.box_iou.zero_()

        if self.eval_generation:
            self.sentence_metrics.reset()
            self._has_sentence_updates = False

    @torch.inference_mode()
    def update(self, model_output: "ImgTextTrainOutput"):
        N = model_output.N

        step_metrics = dict(model_output.step_metrics, loss=model_output.loss.detach())
        self.step_metrics_meter.add(to_device(step_metrics, 'cpu'))

        if model_output.encoded_sentences is not None:
            sent_mask = to_device(model_output.encoded_sentences.sentence_mask, self.device)
            boxes = to_device(model_output.grounding.boxes, self.device)
            self.box_stats.update(boxes, mask=sent_mask)

            
            # recon/l2, recon/cosine, recon/rank
            # (N x S x d)
            sent_features = to_device(model_output.encoded_sentences.sentence_features, self.device)
            # (N x S x d)
            reg_features = to_device(model_output.grounding.box_features, self.device)
            # (N x S)
            l2 = (sent_features - reg_features).norm(p=2.0, dim=-1)
            # (N x S x S_r)
            cos_pairwise = F.normalize(sent_features, dim=-1) @ F.normalize(reg_features, dim=-1).transpose(1, 2)
            # (N x S)
            cos = cos_pairwise.diagonal(dim1=1, dim2=2)
            # (N x S)
            rank = (cos_pairwise > cos.unsqueeze(-1)).sum(dim=-1) + 1
        
            self.count += N
            # (N)
            self.mean_l2 += ((sent_mask * l2).sum(dim=1) / sent_mask.sum(dim=1)).sum()
            self.mean_cos += ((sent_mask * cos).sum(dim=1) / sent_mask.sum(dim=1)).sum()
            self.mean_rank += ((sent_mask * rank).sum(dim=1) / sent_mask.sum(dim=1)).sum()
            
            # box_iou
            # (N x S x S x 4)
            expanded_boxes_1 = einops.repeat(boxes, 'n s1 d -> n s1 s2 d', s2=boxes.shape[1])
            expanded_boxes_2 = einops.repeat(boxes, 'n s2 d -> n s1 s2 d', s1=boxes.shape[1])
            # (N x S x S)
            pairwise_ious = batched_box_iou(expanded_boxes_1, expanded_boxes_2)
            pairwise_ious.diagonal(dim1=1, dim2=2).fill_(0.0)
            pairwise_mask = sent_mask[:, :, None] * sent_mask[:, None, :]
            pairwise_mask.diagonal(dim1=1, dim2=2).fill_(0.0)
            # (N)
            mean_iou = (pairwise_ious * pairwise_mask).sum(dim=(1, 2)) / pairwise_mask.sum(dim=(1, 2)).clamp(min=1e-7)
            self.box_iou += mean_iou.sum()

        # generation
        if self.eval_generation and model_output.generated_sentences is not None and model_output.encoded_sentences is not None:
            self._has_sentence_updates = True
            gen_sentences = model_output.generated_sentences
            target_sentences = model_output.encoded_sentences.sentences
            self.sentence_metrics.update(gen_sentences, target_sentences)
    
    @torch.inference_mode()
    def compute(self):
        metrics = {
            "boxstats/box_iou": (self.box_iou / self.count).cpu(),
            "recon/l2": (self.mean_l2 / self.count).cpu(),
            "recon/cosine": (self.mean_cos / self.count).cpu(),
            "recon/rank": (self.mean_rank / self.count).cpu(),
            **{k: v.cpu() for k, v in self.box_stats.compute().items()},
            **{k: v.cpu() for k, v in self.step_metrics_meter.compute().items()},
        }

        if self.eval_generation and self._has_sentence_updates:
            metrics.update({f"gen/{k}": v for k, v in self.sentence_metrics.compute().items()})

        return metrics
