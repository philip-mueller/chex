from typing import Dict, List, Optional
from torch import nn
import torch
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelMatthewsCorrCoef, MultilabelRankingAveragePrecision, MulticlassMatthewsCorrCoef, MulticlassAUROC, MulticlassAccuracy

from util.data_utils import to_device

class ClassificationMetrics(nn.Module):
    def __init__(self, class_names: List[str], return_class_metrics: bool = False, device='cpu') -> None:
        super().__init__()
        self.class_names = [cls_name.replace(' ', '_').replace('/', '-') for cls_name in class_names]
        
        self.auroc = MultilabelAUROC(num_labels=len(class_names), average='macro')
        self.f1 = MultilabelF1Score(num_labels=len(class_names), average='macro')
        self.mcc_macro = MultilabelMatthewsCorrCoef(num_labels=len(class_names), average='macro')
        self.label_rank_avg_prec = MultilabelRankingAveragePrecision(num_labels=len(class_names))

        self.return_class_metrics = return_class_metrics
        if return_class_metrics:
            self.class_auroc = MultilabelAUROC(num_labels=len(class_names), average='none')

        self.device = device
        self.to(device)

    @torch.inference_mode()
    def update(self, preds, pred_probs, targets):
        preds = to_device(preds, self.device)
        pred_probs = to_device(pred_probs, self.device)
        targets = to_device(targets, self.device)

        self.auroc.update(pred_probs, targets)
        self.f1.update(preds, targets)
        self.mcc_macro.update(preds, targets)
        self.label_rank_avg_prec.update(pred_probs, targets)

        if self.return_class_metrics:
            self.class_auroc.update(pred_probs, targets)

    def reset(self):
        self.auroc.reset()
        self.f1.reset()
        self.mcc_macro.reset()
        self.label_rank_avg_prec.reset()

        if self.return_class_metrics:
            self.class_auroc.reset()

    @torch.inference_mode()
    def compute(self) -> dict:
        metrics = {
            'auroc': self.auroc.compute().cpu(),
            'f1': self.f1.compute().cpu(),
            'mcc_macro': self.mcc_macro.compute().cpu(),
            'label_rank_avg_prec': self.label_rank_avg_prec.compute().cpu(),
        }

        if self.return_class_metrics:
            metrics.update({
                f'auroc_cls/{cls_name}': auroc.cpu()
                for cls_name, auroc in zip(self.class_names, self.class_auroc.compute())
            })

        return metrics
    
class MultiClassClassificationMetrics(nn.Module):
    def __init__(self, class_names: List[str], return_class_metrics: bool = False, device='cpu') -> None:
        super().__init__()
        self.class_names = [cls_name.replace(' ', '_').replace('/', '-') for cls_name in class_names]
        
        self.auroc = MulticlassAUROC(num_classes=len(class_names), average='none' if return_class_metrics else 'macro')
        self.acc = MulticlassAccuracy(num_classes=len(class_names), average='macro')
        self.mcc_macro = MulticlassMatthewsCorrCoef(num_classes=len(class_names), average='macro')

        self.return_class_metrics = return_class_metrics
        self.device = device
        self.to(device)

    @torch.inference_mode()
    def update(self, preds, pred_probs, targets):
        preds = to_device(preds, self.device)
        pred_probs = to_device(pred_probs, self.device)
        targets = to_device(targets, self.device)

        self.auroc.update(pred_probs, targets)
        self.acc.update(preds, targets)
        self.mcc_macro.update(preds, targets)

    def reset(self):
        self.auroc.reset()
        self.acc.reset()
        self.mcc_macro.reset()

    @torch.inference_mode()
    def compute(self) -> dict:
        class_aurocs = self.auroc.compute().cpu()
        macro_auroc = class_aurocs.mean()
        metrics = {
            'auroc': macro_auroc,
            'acc': self.acc.compute().cpu(),
            'mcc': self.mcc_macro.compute().cpu(),
        }

        if self.return_class_metrics:
            metrics.update({
                f'auroc_cls/{cls_name}': cls_auroc
                for cls_name, cls_auroc in zip(self.class_names, class_aurocs)
            })

        return metrics



class RegionMultiLabelClassificationMetrics(nn.Module):
    def __init__(self, anatomy_names: List[str], class_names: List[str], 
                 return_class_metrics: bool = False, return_anatomy_metrics: bool = False,
                 device='cpu') -> None:
        super().__init__()
        self.class_names = [cls_name.replace(' ', '_').replace('/', '-') for cls_name in class_names]
        self.anatomy_names = [anatomy_name.replace(' ', '_').replace('/', '-') for anatomy_name in anatomy_names]
        self.return_class_metrics = return_class_metrics
        self.return_anatomy_metrics = return_anatomy_metrics

        # (A x C)
        self.register_buffer('anat_cls_counts', torch.zeros(len(anatomy_names), len(class_names)))
        self.register_buffer('anat_count', torch.tensor(0))
        self.register_buffer('cls_count', torch.tensor(0))
        self.register_buffer('sample_count', torch.tensor(0))


        self.auroc_anatomy = nn.ModuleList(MultilabelAUROC(num_labels=len(class_names), average='none') for _ in self.anatomy_names)
        self.f1_anatomy = nn.ModuleList(MultilabelF1Score(num_labels=len(class_names), average='none') for _ in self.anatomy_names)

        self.classification_metrics = ClassificationMetrics(class_names, return_class_metrics=False)

        self.device = device
        self.to(device)

    @torch.inference_mode()
    def update(self, anatomy_observation_preds, anatomy_observation_probs, target_anatomy_observations, anatomy_masks: Optional[torch.Tensor] = None):
        """
        anatomy_observation_preds: (N x A x C)
        anatomy_observation_probs: (N x A x C)
        target_anatomy_observations: (N x A x C)
        """
        anatomy_observation_preds = to_device(anatomy_observation_preds, self.device)
        anatomy_observation_probs = to_device(anatomy_observation_probs, self.device)
        target_anatomy_observations = to_device(target_anatomy_observations, self.device)
        anatomy_masks = to_device(anatomy_masks, self.device)

        if anatomy_masks is not None:
            anatomy_observation_preds = anatomy_observation_preds * anatomy_masks.unsqueeze(-1)
            anatomy_observation_probs = anatomy_observation_probs * anatomy_masks.unsqueeze(-1)
            target_anatomy_observations = target_anatomy_observations * anatomy_masks.unsqueeze(-1)

        self.anat_cls_counts += target_anatomy_observations.float().sum(dim=0) # (A x C)
        self.anat_count += target_anatomy_observations.any(dim=2).sum()
        self.cls_count += target_anatomy_observations.any(dim=1).sum()
        self.sample_count += len(target_anatomy_observations)

        for a in range(len(self.anatomy_names)):
            if anatomy_masks is not None:
                self.auroc_anatomy[a].update(anatomy_observation_probs[anatomy_masks[:, a], a], target_anatomy_observations[anatomy_masks[:, a], a])
                self.f1_anatomy[a].update(anatomy_observation_preds[anatomy_masks[:, a], a], target_anatomy_observations[anatomy_masks[:, a], a])
            else:
                self.auroc_anatomy[a].update(anatomy_observation_probs[:, a], target_anatomy_observations[:, a])
                self.f1_anatomy[a].update(anatomy_observation_preds[:, a], target_anatomy_observations[:, a])

        unlocalized_preds = anatomy_observation_preds.any(dim=1) # (N x C)
        unlocalized_probs = anatomy_observation_probs.amax(dim=1) # (N x C)
        unlocalized_targets = target_anatomy_observations.any(dim=1) # (N x C)
        self.classification_metrics.update(unlocalized_preds, unlocalized_probs, unlocalized_targets)

    def reset(self):
        self.anat_cls_counts.zero_()
        self.sample_count.zero_()

        for a in range(len(self.anatomy_names)):
            self.auroc_anatomy[a].reset()
            self.f1_anatomy[a].reset()

        self.classification_metrics.reset()

    @torch.inference_mode()
    def compute(self) -> dict:
        # (A x C)
        auroc_anat_class = torch.stack([auroc.compute() for auroc in self.auroc_anatomy], dim=0) # (A x C)
        f1_anat_class = torch.stack([f1.compute() for f1 in self.f1_anatomy], dim=0) # (A x C)

        # (C)
        auroc_per_class = (self.anat_cls_counts * auroc_anat_class).sum(dim=0) / self.anat_cls_counts.sum(dim=0).clamp_min(1)
        auroc_macro = auroc_per_class.mean()
        auroc_weighted = (self.anat_cls_counts * auroc_anat_class).sum() / self.anat_cls_counts.sum().clamp_min(1)
        f1_per_class = (self.anat_cls_counts * f1_anat_class).sum(dim=0) / self.anat_cls_counts.sum(dim=0).clamp_min(1)
        f1_macro = f1_per_class.mean()

        nonlocalized_metrics = self.classification_metrics.compute()

        pos_anat_per_sample = self.anat_count / self.sample_count
        pos_cls_per_sample = self.cls_count / self.sample_count
        pos_count_per_sample = self.anat_cls_counts.sum() / self.sample_count

        metrics = {
            'loc_auroc/macro': auroc_macro.cpu(),
            'loc_auroc/weighted': auroc_weighted.cpu(),
            'loc_f1/macro': f1_macro.cpu(),
            **{f'nonloc_{k}': v.cpu() for k, v in nonlocalized_metrics.items()},
            'stats/anat_per_sample': pos_anat_per_sample.cpu(),
            'stats/cls_per_sample': pos_cls_per_sample.cpu(),
            'stats/count_per_sample': pos_count_per_sample.cpu(),
        }
        if self.return_class_metrics:
            metrics.update({
                f'loc_auroc_cls/{cls_name}': auroc.cpu()
                for cls_name, auroc in zip(self.class_names, auroc_per_class)
            })

        if self.return_anatomy_metrics:
            auroc_per_anat = (self.anat_cls_counts * auroc_anat_class).sum(dim=1) / self.anat_cls_counts.sum(dim=1).clamp_min(1)
            metrics.update({
                f'loc_auroc_anat/{anat_name}': auroc.cpu()
                for anat_name, auroc in zip(self.anatomy_names, auroc_per_anat)
            })
        
        return metrics
