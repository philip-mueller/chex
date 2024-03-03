from collections import defaultdict
from copy import deepcopy
import csv
import os
import re
import tempfile
from typing import Dict, List, Optional, Union
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from torch import nn
import torch
from metrics.chexbert.constants import CONDITIONS
from metrics.chexbert.label import label
from metrics.chexbert.models.bert_labeler import bert_labeler
from util.data_utils import to_device
import numpy as np
import pandas as pd

import logging
log = logging.getLogger(__name__)

class FullReportMetrics(nn.Module):
    def __init__(self, use_bleu=True, use_rouge=False, use_meteor=True, use_cider=False, use_ce=True, ce_per_class=False) -> None:
        super().__init__()
        self.nlg_micro = NLGMetrics(use_bleu=use_bleu, use_rouge=use_rouge, use_meteor=use_meteor, use_cider=use_cider, micro=True)
        self.use_ce = use_ce
        if use_ce:
            self.ce_metrics = CEMetrics(per_class=ce_per_class, micro=True, macro=True, sample_macro=True)

    def update(self, preds: List[str], targets: List[str]):
        assert len(preds) == len(targets)
        self.nlg_micro.update(preds, targets)
        if self.use_ce:
            self.ce_metrics.update(preds, targets)

    def reset(self):
        self.nlg_micro.reset()
        if self.use_ce:
            self.ce_metrics.reset()

    def compute(self, return_ce_labels=False):
        metrics = {}
        ce_labels = None
        metrics.update(self.nlg_micro.compute())
        if self.use_ce:
            if return_ce_labels:
                ce_metrics, ce_labels = self.ce_metrics.compute(return_labels=True)
                metrics.update(ce_metrics)
            else:
                metrics.update(self.ce_metrics.compute())
        if return_ce_labels:
            return metrics, ce_labels
        else:
            return metrics


class SentenceMetrics(nn.Module):
    def __init__(self, 
                 micro=True, sample_macro=False, region_macro=False, per_region=False, normal_abnormal=False, region_names: Optional[List[str]]=None,
                 use_bleu=True, use_rouge=False, use_meteor=True, use_cider=False, use_ratio=False, use_ce=True, ce_per_class=False):
        super().__init__()

        # ------- NLG metrics -------
        self.micro = micro
        if micro or use_ratio:
            self.nlg_micro = NLGMetrics(use_bleu=use_bleu, use_rouge=use_rouge, use_meteor=use_meteor, use_cider=use_cider, micro=True)
        
        self.sample_macro = sample_macro
        if sample_macro:
            self.nlg_sample_macro = NLGMetrics(use_bleu=use_bleu, use_rouge=use_rouge, use_meteor=use_meteor, use_cider=use_cider, micro=False)
        self.use_ratio = use_ratio
        if use_ratio:
            self.nlg_sample_non_matched = NLGMetrics(use_bleu=use_bleu, use_rouge=use_rouge, use_meteor=use_meteor, use_cider=use_cider, micro=True) # micro=False)
        self.normal_abnormal = normal_abnormal
        if normal_abnormal:
            self.nlg_normal_abnormal = NLGMetrics(use_bleu=use_bleu, use_rouge=use_rouge, use_meteor=use_meteor, use_cider=use_cider, micro=False)
        self.region_macro = region_macro
        self.per_region = per_region
        self.num_regions = None
        if region_macro or per_region:
            self.nlg_per_class = NLGMetrics(use_bleu=use_bleu, use_rouge=use_rouge, use_meteor=use_meteor, use_cider=use_cider, micro=False)
            self.region_names = region_names
            self.num_regions = len(region_names)

        # ------- CE metrics -------
        self.use_ce = use_ce
        if use_ce:
            self.ce_metrics = CEMetrics(per_class=ce_per_class, per_region=per_region, sample_macro=True, micro=True, macro=True)

    def update(self, preds: List[List[str]], targets: List[List[str]], region_ids: List[torch.Tensor]=None, is_normal: List[torch.BoolTensor]=None):  # masks: Optional[torch.Tensor]=None
        assert len(preds) == len(targets)
        pred_lengths = [len(pred_list) for pred_list in preds]
        target_lengths = [len(target_list) for target_list in targets]
        assert pred_lengths == target_lengths, f'pred_lengths: {pred_lengths}, target_lengths: {target_lengths}'
        if region_ids is not None:
            region_ids = [reg_id.cpu() for reg_id in region_ids]
            assert len(preds) == len(region_ids)
            assert all(len(pred_list) == len(reg_id_list) for pred_list, reg_id_list in zip(preds, region_ids)), f'pred_lengths: {pred_lengths}, region_ids: {[len(reg_id_list) for reg_id_list in region_ids]}'
        if is_normal is not None:
            is_normal = [is_normal_i.cpu() for is_normal_i in is_normal]
            assert len(preds) == len(is_normal)
            assert all(len(pred_list) == len(is_normal_list) for pred_list, is_normal_list in zip(preds, is_normal))
            flattened_is_normal = torch.stack([is_normal for is_normal_list in is_normal for is_normal in is_normal_list]).int().numpy().tolist()
        else:
            assert not self.normal_abnormal
        # if masks is not None:
        #     assert masks.ndim == 2
        #     assert masks.shape[0] == len(preds)
        #     assert self.num_regions is None or masks.shape[1] == self.num_regions
        #     assert all(pred_len == mask.shape[0] for pred_len, mask in zip(pred_lengths, masks))
        #     masks = masks.cpu()
        #     preds = [[pred for pred, mask in zip(pred_list, mask_list) if mask] for pred_list, mask_list in zip(preds, masks)]
        #     targets = [[target for target, mask in zip(target_list, mask_list) if mask] for target_list, mask_list in zip(targets, masks)]
            
        flattened_preds = [pred for pred_list in preds for pred in pred_list]
        flattened_targets = [target for target_list in targets for target in target_list]
        sample_ids: List[int] = []
        for i, pred_list in enumerate(preds):
            sample_ids.extend([i] * len(pred_list))
        assert len(flattened_preds) == len(flattened_targets) == len(sample_ids)

        if self.micro or self.use_ratio:
            self.nlg_micro.update(flattened_preds, flattened_targets)
        if self.sample_macro:
            self.nlg_sample_macro.update(flattened_preds, flattened_targets, sample_ids)
        if self.normal_abnormal:
            self.nlg_normal_abnormal.update(flattened_preds, flattened_targets, flattened_is_normal)
        if self.use_ratio:
            flattened_preds_non_matched = []
            flattened_targets_non_matched = []
            #sample_ids_non_matched = []
            for i, (pred_list, target_list) in enumerate(zip(preds, targets)):
                for p, pred_sent in enumerate(pred_list):
                    for t, target_sent in enumerate(target_list):
                        if p == t:
                            continue  # skip "correct" match
                        else:
                            flattened_preds_non_matched.append(pred_sent)
                            flattened_targets_non_matched.append(target_sent)
                            #sample_ids_non_matched.append(i)
            self.nlg_sample_non_matched.update(flattened_preds_non_matched, flattened_targets_non_matched)#, sample_ids_non_matched)

        if self.region_macro or self.per_region:
            if region_ids is not None:
                region_ids = [sample_class_ids.cpu().tolist() for sample_class_ids in region_ids]
                flattened_region_ids = [reg_id for reg_ids_sample in region_ids for reg_id in reg_ids_sample]
            else:
                assert self.num_regions is not None
                assert all(self.num_regions == len(pred_list) for pred_list in preds)
                assert all(self.num_regions == len(target_list) for target_list in targets)
                # if masks is None:
                flattened_region_ids = list(range(self.num_regions)) * len(preds)
                # else:
                #region_ids = [torch.arange(self.num_regions)[mask].cpu().tolist() for mask in masks]
                #flattened_region_ids = [reg_id for reg_ids_sample in region_ids for reg_id in reg_ids_sample]
                

            flattened_region_names = [self.region_names[region_id] for region_id in flattened_region_ids]
            self.nlg_per_class.update(flattened_preds, flattened_targets, flattened_region_ids)
        else:
            flattened_region_names = None

        if self.use_ce:
            self.ce_metrics.update(flattened_preds, flattened_targets, sample_ids, region_names=flattened_region_names)

    def reset(self):
        if self.micro or self.use_ratio:
            self.nlg_micro.reset()
        if self.sample_macro:
            self.nlg_sample_macro.reset()
        if self.use_ratio:
            self.nlg_sample_non_matched.reset()
        if self.region_macro or self.per_region:
            self.nlg_per_class.reset()
        if self.use_ce:
            self.ce_metrics.reset()

    def compute(self):
        metrics = {}
        if self.micro and not self.use_ratio:
            log.info('Computing micro NLG metrics...')
            metrics.update(self.nlg_micro.compute())
        if self.sample_macro:
            log.info('Computing macro NLG metrics...')
            metrics.update(self.nlg_sample_macro.compute(postfix='ex'))
        if self.normal_abnormal:
            log.info('Computing normal/abnormal NLG metrics...')
            metrics.update(self.nlg_normal_abnormal.compute(return_per_ids=True, return_macro=False, per_id_postfix='normal', id_names=['abnormal', 'normal']))
        if self.use_ratio:
            log.info('Computing NLG ratio metrics...')
            # macro_metrics = self.nlg_sample_macro.compute(postfix='ex', return_per_ids=True, per_id_postfix='id', return_per_id_arrays=True)
            # non_matched_metrics = self.nlg_sample_non_matched.compute(postfix='nonmatched', return_per_ids=True, per_id_postfix='id', return_per_id_arrays=True)

            # macro_id_scores = {key.replace('id', 'ratio'): value for key, value in macro_metrics.items() if 'id' in key}
            # non_matched_id_scores = {key.replace('id', 'ratio'): value for key, value in non_matched_metrics.items() if 'id' in key}
            # ratio_id_scores = {
            #     key: (macro_id_scores[key] / non_matched_id_scores[key]).mean()
            #     for key in macro_id_scores.keys()
            # }
            # metrics.update(ratio_id_scores)
            
            # if self.sample_macro:
            #     metrics.update({key: value for key, value in macro_metrics.items() if 'id' not in key})

            micro_metrics = self.nlg_micro.compute()
            non_matched_metrics = self.nlg_sample_non_matched.compute()

            ratio_scores = {
                f'{key}_ratio': (micro_metrics[key] / non_matched_metrics[key]).mean()
                for key in micro_metrics.keys()
            }
            metrics.update(ratio_scores)
            
            if self.micro:
                metrics.update(micro_metrics)
        if self.region_macro or self.per_region:
            log.info('Computing region-wise NLG metrics...')
            metrics.update(self.nlg_per_class.compute(return_per_ids=self.per_region, postfix='regmacro', per_id_postfix='reg', id_names=self.region_names))
        if self.use_ce:
            log.info('Computing CE metrics...')
            metrics.update(self.ce_metrics.compute())
        return metrics


class NLGMetrics(nn.Module):
    def __init__(self, use_bleu=True, use_rouge=True, use_meteor=True, use_cider=False, use_bertscore=False, micro=True, filter_empty_ref=True) -> None:
        super().__init__()

        self.use_bleu = use_bleu
        self.use_rouge = use_rouge
        self.use_meteor = use_meteor
        self.use_cider = use_cider
        self.use_bertscore = use_bertscore
        self.micro = micro
        if micro:
            self.predicted_sentences = []
            self.reference_sentences = []
        else:
            self.predicted_sentences = defaultdict(list)
            self.reference_sentences = defaultdict(list)
            self.ids = set()
        self.filter_empty_ref = filter_empty_ref

    def update(self, preds: List[str], targets: List[str], macro_ids: Optional[List[int]] = None):
        assert len(preds) == len(targets)
        preds = [pred.strip() for pred in preds]
        targets = [target.strip() for target in targets]
        if self.filter_empty_ref:
            if macro_ids is None:
                preds, targets = zip(*[(pred, target) for pred, target in zip(preds, targets) if target != ''])
            else:
                preds, targets, macro_ids = zip(*[(pred, target, macro_id) for pred, target, macro_id in zip(preds, targets, macro_ids) if target != ''])
            if len(preds) == 0:
                return
            
        if self.micro:
            self.predicted_sentences.extend(preds)
            self.reference_sentences.extend(targets)
        else:
            assert macro_ids is not None and len(macro_ids) == len(preds) == len(targets)
            for macro_id, pred, target in zip(macro_ids, preds, targets):
                self.predicted_sentences[macro_id].append(pred)
                self.reference_sentences[macro_id].append(target)
                self.ids.add(macro_id)

    def compute(self, return_per_ids=False, return_macro=True, postfix=None, per_id_postfix='detailed', id_names: Optional[List[str]]=None, return_per_id_arrays=False) -> Dict[str, float]:
        postfix = '' if postfix is None else f'_{postfix}'
        if self.micro:
            assert not return_per_ids
            scores = self._compute_scores(self.predicted_sentences, self.reference_sentences)
            return {f'{metric_name}{postfix}': score for metric_name, score in scores.items()}
        else:
            scores = {
                macro_id: self._compute_scores(self.predicted_sentences[macro_id], self.reference_sentences[macro_id])
                for macro_id in self.ids
            }
            metric_names = list(scores[0].keys())

            macro_scores = {
                metric: sum(score[metric] for score in scores.values()) / len(scores)
                for metric in metric_names
            }
            macro_scores = {f'{metric_name}{postfix}': score for metric_name, score in macro_scores.items()}
            if return_per_ids:
                if return_per_id_arrays:
                    per_id_scores = {
                        f'{metric_name}_{per_id_postfix}': np.array([scores[macro_id][metric_name] for macro_id in self.ids])
                        for metric_name in metric_names
                    }
                else:
                    if id_names is None:
                        id_names = [str(i) for i in range(len(self.ids))]
                    per_id_scores = {
                        f'{metric_name}_{per_id_postfix}/{id_names[macro_id]}': scores[macro_id][metric_name]
                        for metric_name in metric_names
                        for macro_id in self.ids
                    }
                return {**macro_scores, **per_id_scores} if return_macro else per_id_scores
            else:
                assert return_macro
                return macro_scores
            
    def reset(self):
        if self.micro:
            self.predicted_sentences = []
            self.reference_sentences = []
        else:
            self.predicted_sentences = defaultdict(list)
            self.reference_sentences = defaultdict(list)
            self.ids = set()

    def _compute_scores(self, preds: List[str], targets: List[str]) -> Dict[str, float]:
        if len(preds) == 0:
            metrics = {}
            if self.use_bleu:
                metrics['BLEU-1'], metrics['BLEU-2'], metrics['BLEU-3'], metrics['BLEU-4'] = 0.0, 0.0, 0.0, 0.0
            if self.use_rouge:
                metrics['Rouge-L'] = 0.0
            if self.use_meteor:
                metrics['METEOR'] = 0.0
            if self.use_cider:
                metrics['CIDEr'] = 0.0
            if self.use_bertscore:
                metrics['BERTScore/precision'] = 0.0
                metrics['BERTScore/recall'] = 0.0
                metrics['BERTScore/f1'] = 0.0
            return metrics

        preds = convert_for_pycoco_scorer(preds)
        targets = convert_for_pycoco_scorer(targets)

        metrics = {}
        if self.use_bleu:
            from pycocoevalcap.bleu.bleu import Bleu
            score = Bleu(4).compute_score(deepcopy(targets), deepcopy(preds))[0]
            metrics['BLEU-1'], metrics['BLEU-2'], metrics['BLEU-3'], metrics['BLEU-4'] = score
        if self.use_rouge:
            from pycocoevalcap.rouge.rouge import Rouge
            metrics['Rouge-L'] = Rouge().compute_score(deepcopy(targets), deepcopy(preds))[0]
        if self.use_meteor:
            from pycocoevalcap.meteor.meteor import Meteor
            try:
                metrics['METEOR'] = Meteor().compute_score(deepcopy(targets), deepcopy(preds))[0]
            except Exception as e:
                log.error(f'Error in computing METEOR: {e}')
                metrics['METEOR'] = 0.0
        if self.use_cider:
            #from pycocoevalcap.cider.cider import Cider
            #metrics['CIDEr'] = Cider().compute_score(deepcopy(targets), deepcopy(preds))[0]
            from .cider.cider import Cider as CiderMIMIC
            metrics['CIDEr-MIMIC'] = CiderMIMIC().compute_score(deepcopy(targets), deepcopy(preds))[0]
        if self.use_bertscore:
            from torchmetrics.text.bert import BERTScore
            score = BERTScore()(deepcopy(preds), deepcopy(targets))
            metrics.update({
                f'BERTScore/{sub_metric}': score[sub_metric]
                for sub_metric in ['precision', 'recall', 'f1']
            })

        # convert to tensors
        metrics = {k: torch.tensor(v) for k, v in metrics.items()}

        return metrics

class CEMetrics(nn.Module):
    def __init__(self, per_class=False, per_region=False, sample_macro=False, micro=True, macro=True, filter_empty_ref=True) -> None:
        super().__init__()

        self.all_preds = []
        self.all_targets = []
        self.sample_ids = []
        self.region_names = []

        self.per_class = per_class
        self.per_region = per_region
        self.sample_macro = sample_macro
        self.micro = micro
        self.macro = macro
        self.filter_empty_ref = filter_empty_ref

    def update(self, preds: List[str], targets: List[str], sample_ids: List[int] = None, region_names: List[str] = None):
        assert len(preds) == len(targets)
        preds = [pred.strip() for pred in preds]
        targets = [target.strip() for target in targets]
        if self.filter_empty_ref:
            if sample_ids is not None:
                assert len(preds) == len(sample_ids)
                sample_ids = [sample_id for target, sample_id in zip(targets, sample_ids) if target != '']
            if region_names is not None:
                assert len(preds) == len(region_names)
                region_names = [region_name for target, region_name in zip(targets, region_names) if target != '']
            preds, targets = zip(*[(pred, target) for pred, target in zip(preds, targets) if target != ''])
            if len(preds) == 0:
                return
        assert len(preds) == len(targets)
        assert sample_ids is None or len(preds) == len(sample_ids)
            
        if sample_ids is None:
            sample_ids = list(range(len(self.all_preds), len(self.all_preds) + len(preds)))
        self.all_preds.extend(preds)
        self.all_targets.extend(targets)
        self.sample_ids.extend(sample_ids)
        if region_names is not None:
            self.region_names.extend(region_names)

    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.sample_ids = []
        self.region_names = []

    def compute(self, return_labels=False):
        pred_labels, target_labels = self._label(_get_chexbert())
        pred_labels_14, target_labels_14 = _convert_labels_nicolson_14(pred_labels), _convert_labels_nicolson_14(target_labels)
        pred_labels_5, target_labels_5 = _convert_labels_miura_5(pred_labels), _convert_labels_miura_5(target_labels)

        metrics = {}
        if self.micro:
            metrics.update(self._compute_micro_scores(pred_labels_5, target_labels_5, class_names=PATHOLOGIES_5))
            metrics.update(self._compute_micro_scores(pred_labels_14, target_labels_14, class_names=PATHOLOGIES_14))
        if self.macro or self.per_class:
            metrics.update(self._compute_class_macro_scores(pred_labels_5, target_labels_5, class_names=PATHOLOGIES_5))
            metrics.update(self._compute_class_macro_scores(pred_labels_14, target_labels_14, class_names=PATHOLOGIES_14, per_class=self.per_class))
        if self.per_region or self.sample_macro:
            metrics.update(self._compute_sample_and_region_scores(pred_labels_14, target_labels_14, class_names=PATHOLOGIES_14))

        if return_labels:
            return metrics, (pred_labels, target_labels)
        else:
            return metrics

    def _compute_micro_scores(self, pred_labels: np.ndarray, target_labels: np.ndarray, class_names):
        n_cls = len(class_names)
        precision, recall, f1, _ = precision_recall_fscore_support(target_labels.flatten(), pred_labels.flatten(), average="binary")
        return {f'micro_prec_{n_cls}': precision, f'micro_rec_{n_cls}': recall, f'micro_f1_{n_cls}': f1}

    def _compute_class_macro_scores(self, pred_labels: np.ndarray, target_labels: np.ndarray, class_names, per_class=False):
        n_cls = len(class_names)
        precision, recall, f1, _ = precision_recall_fscore_support(target_labels, pred_labels)
        prec_macro, rec_macro, f1_macro = precision.mean(), recall.mean(), f1.mean()
        metrics = {f'macro_prec_{n_cls}': prec_macro, f'macro_rec_{n_cls}': rec_macro, f'macro_f1_{n_cls}': f1_macro}
        if per_class:
            metrics.update({
                f'cls_f1/{class_name}': f1_cls
                for class_name, f1_cls in zip(class_names, f1)
            })
        return metrics

    def _compute_sample_and_region_scores(self, pred_labels: np.ndarray, target_labels: np.ndarray, class_names):
        n_cls = len(class_names)
        MCM = multilabel_confusion_matrix(target_labels, pred_labels, samplewise=True)
        tp = MCM[:, 1, 1]
        fp = MCM[:, 0, 1]
        fn = MCM[:, 1, 0]
        metrics = {}
        if self.sample_macro:
            sample_ids = np.array(self.sample_ids)
            assert sample_ids.shape == tp.shape == fp.shape == fn.shape
            sample_df = pd.DataFrame({'sample_id': sample_ids, 'tp': tp, 'fp': fp, 'fn': fn}).groupby('sample_id').sum()
            sample_df['prec'] = sample_df['tp'] / (sample_df['tp'] + sample_df['fp'])
            sample_df['rec'] = sample_df['tp'] / (sample_df['tp'] + sample_df['fn'])
            sample_df['f1'] = 2 * sample_df['prec'] * sample_df['rec'] / (sample_df['prec'] + sample_df['rec'])
            sample_macro: Dict[str, float] = sample_df[['prec', 'rec', 'f1']].mean().to_dict()
            metrics.update({f'ex_{metric_name}_{n_cls}': score for metric_name, score in sample_macro.items()})
       
        if self.per_region:
            region_df = pd.DataFrame({'region_name': self.region_names, 'tp': tp, 'fp': fp, 'fn': fn}).groupby('region_name').sum()
            region_df['prec'] = region_df['tp'] / (region_df['tp'] + region_df['fp'])
            region_df['rec'] = region_df['tp'] / (region_df['tp'] + region_df['fn'])
            region_df['f1'] = 2 * region_df['prec'] * region_df['rec'] / (region_df['prec'] + region_df['rec'])
            region_macro: Dict[str, float] = region_df[['prec', 'rec', 'f1']].mean().to_dict()
            metrics.update({f'regmacro_{metric_name}_{n_cls}': score for metric_name, score in region_macro.items()})
            if self.per_region:
                per_region_f1 = region_df['f1'].to_dict()
                metrics.update({f'reg_f1_{n_cls}/{region_name}': f1 for region_name, f1 in per_region_f1.items()})
        return metrics

    def _label(self, chexbert):
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_file = os.path.join(temp_dir, "reports_pred.csv")
            target_file = os.path.join(temp_dir, "report_target.csv")

            header = ["Report Impression"]
            with open(pred_file, "w") as fp:
                csv_writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
                csv_writer.writerow(header)
                csv_writer.writerows([[gen_report] for gen_report in self.all_preds])
            
            with open(target_file, "w") as fp:
                csv_writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
                csv_writer.writerow(header)
                csv_writer.writerows([[ref_report] for ref_report in self.all_targets])

            # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
            preds_gen_reports: List[List[int]] = label(chexbert, pred_file)
            preds_ref_reports: List[List[int]] = label(chexbert, target_file)
        preds_gen_reports = np.array(preds_gen_reports, dtype=np.int32).T
        preds_ref_reports = np.array(preds_ref_reports, dtype=np.int32).T
        # handle empty sentences (those are considered as no-finding)
        LABEL_FOR_EMPTY = np.zeros(14, dtype=np.int32)
        LABEL_FOR_EMPTY[-1] = 1  # only no finding is positive
        empty_preds = np.array([gen_report.strip() == '' for gen_report in self.all_preds])
        preds_gen_reports[empty_preds] = LABEL_FOR_EMPTY[None, :]
        empty_targets = np.array([ref_report.strip() == '' for ref_report in self.all_targets])
        preds_ref_reports[empty_targets] = LABEL_FOR_EMPTY[None, :]

        return preds_gen_reports, preds_ref_reports


def convert_for_pycoco_scorer(sents_or_reports: List[str]):
    """
    The compute_score methods of the scorer objects require the input not to be list[str],
    but of the form:
    generated_reports =
    {
        "image_id_0" = ["1st generated report"],
        "image_id_1" = ["2nd generated report"],
        ...
    }
    Hence we convert the generated/reference sentences/reports into the appropriate format and also tokenize them
    following Nicolson's (https://arxiv.org/pdf/2201.09405.pdf) implementation (https://github.com/aehrc/cvt2distilgpt2/blob/main/transmodal/metrics/chen.py):
    see lines 132 and 133
    """
    sents_or_reports_converted = {}
    for num, text in enumerate(sents_or_reports):
        sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " .").replace('\n', ' ').replace('\t', ' ').replace('\r', ''))]

    return sents_or_reports_converted

PATHOLOGIES_14 = CONDITIONS
PATHOLOGIES_5 = {"Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"}
_FILTER_5 = np.array([True if class_name in PATHOLOGIES_5 else False for class_name in PATHOLOGIES_14])

def _convert_labels_miura_5(preds_reports: np.ndarray):
    """
    Miura et. al. (https://arxiv.org/pdf/2010.10042.pdf)
    Considers the negative and blank/NaN to be one whole negative class, and positive and uncertain to be one whole positive class.
    For reference, see lines 141 and 143 of Miura's implementation: https://github.com/ysmiura/ifcc/blob/master/eval_prf.py#L141,
    where label 3 is converted to label 1, and label 2 is converted to label 0.
    """
    preds_reports = preds_reports.copy()
    preds_reports = preds_reports[:, _FILTER_5]
    preds_reports[preds_reports == 2] = 0
    preds_reports[preds_reports == 3] = 1
    
    return preds_reports.astype(bool)

def _convert_labels_nicolson_14(preds_reports: list[list[int]]):
    """
    Convert label 1 to True and everything else (i.e. labels 0, 2, 3) to False
    effectively doing the label conversion as done by Nicolson (https://arxiv.org/pdf/2201.09405.pdf, https://github.com/aehrc/cvt2distilgpt2/blob/main/transmodal/metrics/chen.py)
        
    """
    preds_reports = deepcopy(preds_reports)
    preds_reports = preds_reports == 1  # 1 is positive, other is negative

    return preds_reports

def _get_chexbert():
        model = bert_labeler()
        model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
        checkpoint = torch.load(os.path.expanduser("~/models/third_party/chexbert/chexbert.pth"), map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.cuda()
        model.eval()

        return model
