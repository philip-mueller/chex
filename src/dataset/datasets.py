from abc import abstractmethod
import ast
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import accumulate, repeat
import json
import logging
from multiprocessing import Pool
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from omegaconf import MISSING
import pandas as pd

import albumentations as A
import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset, default_collate


import tqdm

from dataset.image_transform import TransformConfig, build_transform
from dataset.mimic_cxr_datasets import load_mimic_cxr_datafile
from dataset.nih_cxr import load_cxr14_dataset
from dataset.vindr_cxr import VINDR_CXR_DATASET_NAME, load_vindr_cxr_datafile
from util.data_utils import load_pil_gray

log = logging.getLogger(__name__)
    

@dataclass
class DatasetConfig:
    name: str = MISSING

    # --- Normalization parameters ---
    pixel_mean: List[float] = MISSING
    pixel_std: List[float] = MISSING

    # --- Oversampling (only relevant for multidataset) ---
    oversampling_factor: float = 1.0
    auto_balance: bool = False

    # --- Annotations / Modalities ---
    class_names: Optional[List[str]] = None
    class_names_with_boxes: Optional[List[str]] = None
    anatomy_names: Optional[List[str]] = None
    multi_anatomy_names: Optional[List[str]] = None

    has_sentences: bool = False
    has_class_labels: bool = False
    has_anatomy_class_labels: bool = False
    has_class_bboxes: bool = False
    has_class_box_sentences: bool = False
    has_anatomy_bboxes: bool = False
    has_anatomy_sentences: bool = False

    uncertain_pos: bool = True
    anatomy_multibox_mapping: Dict[str, List[str]] = field(default_factory=dict)

    load_memmap: bool = True
    limit_samples: Optional[int] = None
    load_images: bool = True

class ImageWithSupervisionDataset(Dataset):
    def __init__(self,
                 config: DatasetConfig,
                 mode: str,
                 image_size,
                 transform: Callable,
                 prefetch: bool = False) -> None:
        super().__init__()

        self.dataset_info = config
        self.dataset_name = config.name
        self.transform = transform

        # --- Load the dataset metadata ---
        if config.name.startswith('mimic_cxr-'):
            data_df, self.load_indices, self.load_fn, loaded_size = load_mimic_cxr_datafile(config.name, mode, image_size=image_size, load_memmap=config.load_memmap)
        elif config.name == VINDR_CXR_DATASET_NAME:
            data_df, self.load_indices, self.load_fn, loaded_size = load_vindr_cxr_datafile(mode, image_size=image_size, load_memmap=config.load_memmap)
        elif config.name == 'nih_cxr':
            data_df, self.load_indices, self.load_fn = load_cxr14_dataset(mode)
            loaded_size = None
        else:
            raise ValueError(f'Unknwon dataset {config.name}. Config: {config}')
        self.sample_ids = data_df.sample_id.to_list()
        
        # images may have already been preprocessed (resized/cropped/padded) but the bboxes are still in the original image coordinates
        # -> we need to compute the crops and rescales to later transform the bboxes to the loaded image coordinates
        if config.has_anatomy_bboxes or config.has_class_bboxes:
            # (N x 2), (N)
            crops, rescales = _compute_crops_and_rescales(data_df, center_cropped=loaded_size is not None)

        # prefetch the images
        if prefetch and config.load_images:
            self.load_indices, self.load_fn = prefetch_wrapper(self.load_indices, self.load_fn)

        
        # --- Load supervision / additional modalities ---
        if config.has_sentences:
            # outer list: one element per sample, inner list: one element per sentence of the sample
            self.sentences: List[List[str]] = load_sentences(data_df, column_name='report_sentences')
            
        if config.has_class_labels:
            # (N x C)
            self.class_labels: np.ndarray = load_class_labels(data_df, config, column_prefix='cls')

        if config.has_anatomy_class_labels:
            # (N x A x C)
            self.anatomy_class_labels: np.ndarray = load_anatomy_class_labels(data_df, config, column_prefix='anatcls')

        if config.has_anatomy_bboxes:
            # (N x A x 4), (N x A); box format: "(x_c, y_c, w, h)" in relative coordinates [0, 1]
            self.anatomy_bboxes, self.anatomy_present_masks = load_anatomy_bboxes(data_df, config, crops=crops, rescales=rescales, column_prefix='anatbbox')
            # (N x A_multi x M x 4), (N x A_multi x M); box format: "(x_c, y_c, w, h)" in relative coordinates [0, 1]
            self.anatomy_multiboxes, self.anatomy_multibox_masks = load_anatomy_multiboxes(self.anatomy_bboxes, self.anatomy_present_masks, config)

        if config.has_class_bboxes:
            assert config.class_names_with_boxes is not None and len(config.class_names_with_boxes) > 0
            # list with one element per sample, each element is a (M_i x 5) array with box format: "(x_c, y_c, w, h, class_id)" in relative coordinates [0, 1]
            self.class_bboxes: List[np.ndarray] = load_class_bboxes(data_df, config, crops=crops, rescales=rescales, column_name='clsbbox')
            self.has_class_bboxes = np.array([c in config.class_names_with_boxes for c in config.class_names], dtype=np.bool8)

        if config.has_anatomy_sentences:
            # outer list: one element per sample, inner list: one element per anatomy regions, inner inner list: one element per sentence of the region
            self.anatomy_sentences: List[List[List[str]]] = load_anatomy_sentences(data_df, config, column_prefix='anatphrases')

        if config.has_class_box_sentences:
            assert config.has_class_bboxes
            self.class_bbox_sentences: List[List[str]] = load_class_box_sentences(data_df, class_bboxes=self.class_bboxes, column_name='clsbboxphrases')
            self.grounded_sentences, self.sentence_boxes = load_grounded_sentences(self.class_bbox_sentences, self.class_bboxes)

    def __len__(self) -> int:
        assert len(self.sample_ids) > 0, f'No samples found for dataset {self.dataset_name}'
        if self.dataset_info.limit_samples is not None:
            return min(len(self.sample_ids), self.dataset_info.limit_samples)
        else:
            return len(self.sample_ids)

    def __getitem__(self, index) -> dict:
        sample = {
            'sample_id': str(self.sample_ids[index]),
        }
        if not self.dataset_info.load_images:
            if self.dataset_info.has_sentences:
                sample['sentences'] = self.sentences[index]
            return sample

        try:
            img = load_image(self.load_indices, self.load_fn, index)
            if isinstance(img, Image.Image):
                img = np.array(img, dtype=np.float32) / 255.

            # --- transform image (and bboxes) ---
            if self.dataset_info.has_anatomy_bboxes:
                assert not self.dataset_info.has_class_bboxes
                anatomy_boxes = self.anatomy_bboxes[index]  # # (A x 4) of type np.float32 in format (xc, yc, w, h, class_id)
                anatomy_present_masks = self.anatomy_present_masks[index] # (A) of type np.bool
                img, anatomy_boxes, anatomy_present_masks = self._transform_with_anat_boxes(img, anatomy_boxes, anatomy_present_masks)
                sample['target_anat_present_masks'] = anatomy_present_masks
                sample['target_anat_boxes'] = anatomy_boxes
                if self.anatomy_multiboxes is not None:
                    sample['target_anat_multiboxes'] = self.anatomy_multiboxes[index]
                    sample['target_anat_multiboxes_masks'] = self.anatomy_multibox_masks[index]
            elif self.dataset_info.has_class_bboxes:
                assert not self.dataset_info.has_anatomy_bboxes
                class_boxes = self.class_bboxes[index]  # (M x 5) of type np.float32 in format (xc, yc, w, h, class_id)
                img, class_boxes = self._transform_with_class_boxes(img, class_boxes)
                sample['target_cls_boxes'] = class_boxes
                sample['has_class_bboxes'] = self.has_class_bboxes
            else:
                img = self._transform(img)
            sample['x'] = img

            if self.dataset_info.has_class_labels:
                sample['target_cls_labels'] = self.class_labels[index]
            if self.dataset_info.has_anatomy_class_labels:
                sample['target_anat_cls_labels'] = self.anatomy_class_labels[index]
            if self.dataset_info.has_sentences:
                sample['sentences'] = self.sentences[index]
            if self.dataset_info.has_anatomy_sentences:
                sample['target_anat_sentences'] = self.anatomy_sentences[index]
            if self.dataset_info.has_class_box_sentences:
                sample['target_cls_box_sentences'] = self.class_bbox_sentences[index]
                sample['target_sentence_boxes'] = self.sentence_boxes[index]
                sample['grounded_sentences'] = self.grounded_sentences[index]
            
            return sample
        except IndexError as e:
            raise RuntimeError(f'Index {index} out of range for dataset {self.dataset_name} with {len(self)} samples') from e
    
    
    def _transform(self, img):
        return self.transform(image=img, bboxes=[], labels=[])['image']

    def _transform_with_anat_boxes(self, img, anat_bboxes, anat_present_masks):
        A = anat_bboxes.shape[0]
        box_label_mapping = np.arange(A, dtype=np.int64)
        # (n_present_anat_boxes) of type np.int64
        present_labels = box_label_mapping[anat_present_masks]
        # (n_present_anat_boxes x 4) of type np.float32 in format (xc, yc, w, h)
        present_bboxes = anat_bboxes[anat_present_masks, :]

        transformed = self.transform(image=img, bboxes=present_bboxes, labels=present_labels)
        img = transformed['image']

        present_labels = np.array(transformed['labels']).astype(np.int64)
        present_bboxes = np.array(transformed['bboxes'])
        anat_present_masks = np.zeros_like(anat_present_masks)
        anat_bboxes = np.zeros_like(anat_bboxes)
        
        if len(present_labels) > 0:
            anat_present_masks[present_labels] = True
            #assert anat_present_masks.sum() > 0, f'No anatomy boxes remaining'
            anat_bboxes[present_labels, :] = present_bboxes

        return img, anat_bboxes, anat_present_masks
        
    def _transform_with_class_boxes(self, img, class_bboxes):
        M = class_bboxes.shape[0]
        # (M) of type np.int64
        bbox_labels = class_bboxes[:, 4].astype(np.int64)
        class_bboxes = class_bboxes[:, :4]
        transformed = self.transform(image=img, bboxes=class_bboxes, labels=bbox_labels)
        img = transformed['image']

        bbox_labels = np.array(transformed['labels']).astype(np.int64)
        class_bboxes = np.array(transformed['bboxes'])
        M = class_bboxes.shape[0]
        class_bboxes = np.concatenate([class_bboxes, bbox_labels[:, None]], axis=1) if M > 0 else np.zeros((0, 5), dtype=np.float32)

        return img, class_bboxes

@dataclass
class MultiDatasetInfo:
    names: List[str] = MISSING
    name: str = MISSING

    # --- Subset Configs ---
    subset_infos: List[DatasetConfig] = MISSING
    subset_anatomy_ranges: List[Optional[Tuple[int, int]]] = MISSING
    subset_multianatomy_ranges: List[Optional[Tuple[int, int]]] = MISSING
    subset_class_ranges: List[Optional[Tuple[int, int]]] = MISSING

    # --- Combined ---
    class_names: Optional[List[str]] = None
    anatomy_names: Optional[List[str]] = None
    multi_anatomy_names: Optional[List[str]] = None

    has_sentences: bool = False
    has_class_labels: bool = False
    has_anatomy_class_labels: bool = False
    has_class_bboxes: bool = False
    has_class_box_sentences: bool = False
    has_anatomy_bboxes: bool = False
    has_anatomy_sentences: bool = False


class MultiDataset(Dataset):
    def __init__(self, datasets: List[ImageWithSupervisionDataset]) -> None:
        super().__init__()
        self.datasets = datasets
        oversampling_factors = [dataset.dataset_info.oversampling_factor for dataset in datasets]
        auto_balance = [dataset.dataset_info.auto_balance for dataset in datasets]
        dataset_sizes = [len(dataset) for dataset in datasets]
        if any(auto_balance):
            max_size = max(dataset_sizes)
            auto_oversampling_factors = [max_size / size for size in dataset_sizes]
            auto_oversampling_factors = [factor if automatic else 1 for factor, automatic in zip(auto_oversampling_factors, auto_balance)]
            oversampling_factors = [auto_factor * manual_factor for auto_factor, manual_factor in zip(auto_oversampling_factors, oversampling_factors)]
        oversampling_factors = [int(round(factor)) for factor in oversampling_factors]
        self.oversampling_factors = oversampling_factors
        self.effective_dataset_sizes = [oversampling_factor * dataset_size for oversampling_factor, dataset_size in zip(oversampling_factors, dataset_sizes)]
        self.dataset_sizes = dataset_sizes
        self.size = sum(self.effective_dataset_sizes)

        self.dataset_indices = [i_dataset for i_dataset, len_dataset in zip(range(len(datasets)), self.effective_dataset_sizes) for _ in range(len_dataset)]
        self.dataset_starts = np.cumsum([0] + self.effective_dataset_sizes[:-1])
        
        self.dataset_info = MultiDatasetInfo(
            names=[dataset.dataset_info.name for dataset in datasets], 
            name='+'.join([dataset.dataset_info.name for dataset in datasets]),
            subset_infos=[dataset.dataset_info for dataset in datasets])
        self.dataset_info.subset_anatomy_ranges = [None for _ in datasets]
        self.dataset_info.subset_multianatomy_ranges = [None for _ in datasets]
        self.dataset_info.subset_class_ranges = [None for _ in datasets]
        a_index = 0
        a_multi_index = 0
        c_index = 0
        m_multibox_list = [dataset.anatomy_multiboxes.shape[2] for dataset in datasets if dataset.dataset_info.has_anatomy_bboxes and dataset.anatomy_multiboxes is not None]
        self.m_multibox = max(m_multibox_list) if len(m_multibox_list) > 0 else 0
        for i_dataset, subset_info in enumerate(self.dataset_info.subset_infos):
            if subset_info.class_names is not None and len(subset_info.class_names) > 0:
                assert self.dataset_info.class_names is None or all(cls_name not in self.dataset_info.class_names for cls_name in subset_info.class_names)
                self.dataset_info.class_names = subset_info.class_names if self.dataset_info.class_names is None else self.dataset_info.class_names + subset_info.class_names
                self.dataset_info.subset_class_ranges[i_dataset] = (c_index, c_index + len(subset_info.class_names))
                c_index += len(subset_info.class_names)
            if subset_info.anatomy_names is not None:
                assert self.dataset_info.anatomy_names is None or all(anat_name not in self.dataset_info.anatomy_names for anat_name in subset_info.anatomy_names)
                self.dataset_info.anatomy_names = subset_info.anatomy_names if self.dataset_info.anatomy_names is None else self.dataset_info.anatomy_names + subset_info.anatomy_names
                self.dataset_info.subset_anatomy_ranges[i_dataset] = (a_index, a_index + len(subset_info.anatomy_names))
                a_index += len(subset_info.anatomy_names)
            if subset_info.multi_anatomy_names is not None:
                assert self.dataset_info.multi_anatomy_names is None or all(anat_name not in self.dataset_info.multi_anatomy_names for anat_name in subset_info.multi_anatomy_names)
                self.dataset_info.multi_anatomy_names = subset_info.multi_anatomy_names if self.dataset_info.multi_anatomy_names is None else self.dataset_info.multi_anatomy_names + subset_info.multi_anatomy_names
                self.dataset_info.subset_multianatomy_ranges[i_dataset] = (a_multi_index, a_multi_index + len(subset_info.multi_anatomy_names))
                a_multi_index += len(subset_info.multi_anatomy_names)

            self.dataset_info.has_sentences |= subset_info.has_sentences
            self.dataset_info.has_class_labels |= subset_info.has_class_labels
            self.dataset_info.has_anatomy_class_labels |= subset_info.has_anatomy_class_labels
            self.dataset_info.has_class_bboxes |= subset_info.has_class_bboxes
            self.dataset_info.has_class_box_sentences |= subset_info.has_class_box_sentences
            self.dataset_info.has_anatomy_bboxes |= subset_info.has_anatomy_bboxes
            self.dataset_info.has_anatomy_sentences |= subset_info.has_anatomy_sentences
        
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        i_dataset = self.dataset_indices[index]
        index_in_dataset = index - self.dataset_starts[i_dataset]
        dataset_size = self.dataset_sizes[i_dataset]
        index_in_dataset = index_in_dataset % dataset_size

        sample = self.datasets[i_dataset][index_in_dataset]
        sample['dataset_index'] = i_dataset
        sample['dataset_name'] = self.dataset_info.names[i_dataset]

        return self._fill_empty_targets(sample, i_dataset)
    
    def _fill_empty_targets(self, sample, i_dataset):
        subset_info = self.dataset_info.subset_infos[i_dataset]
        # C/A: number of classes/anatomy of the whole dataset (not just the subset)
        A = len(self.dataset_info.anatomy_names) if self.dataset_info.anatomy_names is not None else 0
        A_multi = len(self.dataset_info.multi_anatomy_names) if self.dataset_info.multi_anatomy_names is not None else 0
        M_multi = self.m_multibox
        C = len(self.dataset_info.class_names) if self.dataset_info.class_names is not None else 0

        if self.dataset_info.has_anatomy_bboxes:
            has_anatomy_bboxes = np.zeros((A,), dtype=np.bool8)
            target_anat_present_masks = np.zeros((A,), dtype=np.bool8)
            target_anat_boxes = np.zeros((A, 4), dtype=np.float32)
            if subset_info.has_anatomy_bboxes:
                a_start, a_end = self.dataset_info.subset_anatomy_ranges[i_dataset]
                has_anatomy_bboxes[a_start:a_end] = True
                target_anat_present_masks[a_start:a_end] = sample['target_anat_present_masks']
                target_anat_boxes[a_start:a_end] = sample['target_anat_boxes']
            sample['has_anatomy_bboxes'] = has_anatomy_bboxes
            sample['target_anat_present_masks'] = target_anat_present_masks
            sample['target_anat_boxes'] = target_anat_boxes

            if self.dataset_info.multi_anatomy_names is not None and len(self.dataset_info.multi_anatomy_names) > 0:
                has_anatomy_multiboxes = np.zeros((A_multi,), dtype=np.bool8)
                target_anat_multiboxes = np.zeros((A_multi, M_multi, 4), dtype=np.float32)
                target_anat_multibox_masks = np.zeros((A_multi, M_multi), dtype=np.bool8)
                if subset_info.has_anatomy_bboxes and subset_info.multi_anatomy_names is not None and len(subset_info.multi_anatomy_names) > 0:
                    am_start, am_end = self.dataset_info.subset_multianatomy_ranges[i_dataset]
                    target_anat_multiboxes[am_start:am_end] = sample['target_anat_multiboxes']
                    target_anat_multibox_masks[am_start:am_end] = sample['target_anat_multiboxes_masks']
                    has_anatomy_multiboxes[am_start:am_end] = True
                sample['target_anat_multiboxes'] = target_anat_multiboxes
                sample['target_anat_multiboxes_masks'] = target_anat_multibox_masks
                sample['has_anatomy_multiboxes'] = has_anatomy_multiboxes

        if self.dataset_info.has_class_bboxes:
            has_class_bboxes = np.zeros((C,), dtype=np.bool8)
            if subset_info.has_class_bboxes:
                c_start, c_end = self.dataset_info.subset_class_ranges[i_dataset]
                has_class_bboxes[c_start:c_end] = sample['has_class_bboxes'] # True
                target_cls_boxes = sample['target_cls_boxes']
                target_cls_boxes[:, 4] = target_cls_boxes[:, 4] + c_start
            else:
                target_cls_boxes = np.zeros((0, 5), dtype=np.float32)
            sample['has_class_bboxes'] = has_class_bboxes
            sample['target_cls_boxes'] = target_cls_boxes

        if self.dataset_info.has_class_labels:
            has_class_labels = np.zeros((C,), dtype=np.bool8)
            target_cls_labels = np.zeros((C,), dtype=np.int64)
            if subset_info.has_class_labels:
                c_start, c_end = self.dataset_info.subset_class_ranges[i_dataset]
                has_class_labels[c_start:c_end] = True
                target_cls_labels[c_start:c_end] = sample['target_cls_labels']
            sample['has_class_labels'] = has_class_labels
            sample['target_cls_labels'] = target_cls_labels

        if self.dataset_info.has_anatomy_class_labels:
            has_anatomy_class_labels = np.zeros((A, C), dtype=np.bool8)
            target_anat_cls_labels = np.zeros((A, C), dtype=np.int64)
            if subset_info.has_anatomy_class_labels:
                a_start, a_end = self.dataset_info.subset_anatomy_ranges[i_dataset]
                c_start, c_end = self.dataset_info.subset_class_ranges[i_dataset]
                has_anatomy_class_labels[a_start:a_end, c_start:c_end] = True
                target_anat_cls_labels[a_start:a_end, c_start:c_end] = sample['target_anat_cls_labels']
            sample['has_anatomy_class_labels'] = has_anatomy_class_labels
            sample['target_anat_cls_labels'] = target_anat_cls_labels

        if self.dataset_info.has_sentences:
            if subset_info.has_sentences:
                has_sentences = np.array(True, dtype=np.bool8)
                target_sentences = sample['sentences']
            else:
                has_sentences = np.array(False, dtype=np.bool8)
                target_sentences = None
            sample['has_sentences'] = has_sentences
            sample['sentences'] = target_sentences

        if self.dataset_info.has_anatomy_sentences:
            has_anatomy_sentences = np.zeros((A,), dtype=np.bool8)
            target_anat_sentences = [[] for _ in range(A)]
            if subset_info.has_anatomy_sentences:
                a_start, a_end = self.dataset_info.subset_anatomy_ranges[i_dataset]
                has_anatomy_sentences[a_start:a_end] = True
                target_anat_sentences[a_start:a_end] = sample['target_anat_sentences']
            sample['has_anatomy_sentences'] = has_anatomy_sentences
            sample['target_anat_sentences'] = target_anat_sentences
        
        if self.dataset_info.has_class_box_sentences:
            has_class_box_sentences = np.zeros((C,), dtype=np.bool8)
            if subset_info.has_class_box_sentences:
                c_start, c_end = self.dataset_info.subset_class_ranges[i_dataset]
                has_class_box_sentences[c_start:c_end] = True
                target_cls_box_sentences = sample['target_cls_box_sentences']
                target_sentence_boxes = sample['target_sentence_boxes']
                grounded_sentences = sample['grounded_sentences']
            else:
                target_cls_box_sentences = []
                target_sentence_boxes = []
                grounded_sentences = []
            sample['has_class_box_sentences'] = has_class_box_sentences
            sample['target_cls_box_sentences'] = target_cls_box_sentences
            sample['target_sentence_boxes'] = target_sentence_boxes
            sample['grounded_sentences'] = grounded_sentences

        return sample
        

def build_dataloader(configs: List[DatasetConfig],
                     mode: str, 
                     pixel_mean: Tuple[float, float, float],
                     pixel_std: Tuple[float, float, float],
                     transform: TransformConfig,
                     batch_size: int,
                     num_workers: int = 0,
                     prefetch: bool = False) -> DataLoader:
    configs = [config if isinstance(config, DatasetConfig) else DatasetConfig(**config) for config in configs]
    if mode == 'train':
        is_train = True
    else:
        assert mode in ('val', 'test')
        is_train = False

    image_size = transform.train_size
    if not is_train and transform.val_size is not None:
        image_size = transform.val_size

    transform = build_transform(
                    transform,
                    is_train=is_train,
                    pixel_mean=pixel_mean, pixel_std=pixel_std)

    assert len(configs) > 0
    if len(configs) == 1:
        dataset = ImageWithSupervisionDataset(
                    configs[0], 
                    mode=mode,
                    image_size=image_size,
                    prefetch=prefetch, 
                    transform=transform)
    else:
        dataset = MultiDataset(
                    [ImageWithSupervisionDataset(
                        conf, 
                        mode=mode,
                        image_size=image_size,
                        prefetch=prefetch, 
                        transform=transform)
                    for conf in configs])
    dataset_info = dataset.dataset_info
    
    # Get one iter of train_ds and val_ds for debugging
    try:
        next(iter(dataset))
    except StopIteration:
        raise RuntimeError(f'No data found in {dataset_info.name} dataset')

    # handle stacking of bounding boxes
    def collate_fn(batch: List[dict]):
        collated_batch = {}

        if dataset_info.has_sentences:
            collated_batch['sentences'] = [sample.pop('sentences') for sample in batch]
        if dataset_info.has_anatomy_sentences:
            collated_batch['target_anat_sentences'] = [sample.pop('target_anat_sentences') for sample in batch]
        if dataset_info.has_class_box_sentences:
            collated_batch['target_cls_box_sentences'] = [sample.pop('target_cls_box_sentences') for sample in batch]
            collated_batch['target_sentence_boxes'] = [[torch.tensor(boxes) for boxes in sample.pop('target_sentence_boxes')] for sample in batch]
            collated_batch['grounded_sentences'] = [sample.pop('grounded_sentences') for sample in batch]
        if dataset_info.has_class_bboxes:
            collated_batch['target_cls_boxes'] = [torch.tensor(sample.pop('target_cls_boxes')) for sample in batch]
        
        collated_batch.update(default_collate(batch))

        if dataset_info.has_class_bboxes:
            class_ids_with_boxes, boxes, box_masks = convert_bbox_list_to_padded_tensor(collated_batch['target_cls_boxes'], C=len(dataset_info.class_names), has_class_bboxes=collated_batch.get('has_class_bboxes'))
            collated_batch['class_ids_with_boxes'] = class_ids_with_boxes
            collated_batch['target_cls_boxes_padded'] = boxes
            collated_batch['target_cls_boxes_mask'] = box_masks

        return collated_batch

    #def seed_worker(worker_id):
    #    worker_seed = torch.initial_seed() % 2**32
    #    np.random.seed(worker_seed)
    #    random.seed(worker_seed)

    #g = torch.Generator()
    #g.manual_seed(0)

    # Create dataloader
    return DataLoader(dataset,
                      batch_size,
                      shuffle=is_train,
                      drop_last=is_train,
                      pin_memory=True,
                      num_workers=num_workers,
                      prefetch_factor=3,
                      collate_fn=collate_fn,
                      #worker_init_fn=seed_worker,
                      #generator=g,
                      persistent_workers=False)



# ============================== Load Labels ============================== #
# -----------> Sentences <----------- #
def load_sentences(data_df, column_name: str) -> List[List[str]]:
    assert column_name in data_df.columns, f'Column {column_name} not found in data_df'
    sentences = data_df[column_name].to_list()
    return  [_parse_sentences(s) for s in sentences]

def load_class_box_sentences(data_df, class_bboxes, column_name: str) -> List[np.ndarray]:
    # --- Load the observation bounding boxes, possible from several observation sets ---
    all_sentences: List[str] = data_df[column_name].fillna('[]').to_list()
    all_sentences: List[List[str]] = [_parse_sentences(sampe_sentences) for sampe_sentences in all_sentences]
    M_is = np.array([len(sampe_sentences) for sampe_sentences in all_sentences])
    M_min, M_max = M_is.min(), M_is.max()
    M_zero = (M_is == 0).sum()
    classboxes_M_is = np.array([len(class_bboxes[i]) for i in range(len(class_bboxes))])
    assert (M_is == classboxes_M_is).all(), f'Number of sentences does not match number of class boxes in some samples'
    log.info(f'Loaded {len(all_sentences)} samples with {M_min} to {M_max} sentences per sample (zero sentences: {M_zero}/{len(all_sentences)})')
    return all_sentences
 
def load_grounded_sentences(class_bbox_sentences: List[List[str]], class_bboxes: List[np.ndarray]) -> Tuple[List[List[np.ndarray]], List[List[str]]]:
    sentence_boxes: List[List[np.ndarray]] = []
    grounded_sentences: List[List[str]] = []
    boxes_per_sentence = []
    for sample_sentences, sample_bboxes in zip(class_bbox_sentences, class_bboxes):
        assert len(sample_sentences) == len(sample_bboxes), f'Number of sentences does not match number of class boxes in some samples ({len(sample_sentences)} vs {len(sample_bboxes)})'

        unique_sample_sentences: List[str] = sorted(set(sample_sentences))
        boxes_by_sentence: Dict[str, List[np.ndarray]] = defaultdict(list)
        #classes_by_sentence: Dict[str, int] = {}
        for sentence, box in zip(sample_sentences, sample_bboxes):
            cls_id = int(box[4])
            #box = box[:4]
            boxes_by_sentence[sentence].append(box)
        # List of length S_i of arrays of shape (M_is, 5)
        sample_sentence_boxes: List[np.ndarray] = [np.stack(boxes_by_sentence[sentence], axis=0) for sentence in unique_sample_sentences]

        sentence_boxes.append(sample_sentence_boxes)
        grounded_sentences.append(unique_sample_sentences)
        boxes_per_sentence.append(float(len(sample_bboxes)) / len(unique_sample_sentences))
    boxes_per_sentence = np.array(boxes_per_sentence).mean()
    log.info(f'Loaded {len(sentence_boxes)} samples with grounded sentences (avg. {boxes_per_sentence:.2f} boxes per sentence)')

    return grounded_sentences, sentence_boxes

def load_anatomy_sentences(data_df, config: DatasetConfig, column_prefix: str) -> List[List[List[str]]]:
    assert config.anatomy_names is not None and len(config.anatomy_names) > 0
    n_samples = len(data_df)
    all_anatomy_sentences: List[List[List[str]]] = [[] for _ in range(n_samples)]
    for a, anatomy_name in enumerate(config.anatomy_names):
        col_name = f'{column_prefix}/{anatomy_name}'
        assert col_name in data_df.columns, f'Column {col_name} not found in data_df. Available columns: {data_df.columns}'
        # len = n_samples, each element is a list of sentences for the anatomy a (in string format)
        anatomy_sentences = data_df[col_name].fillna('[]').to_list()
        for i, sample_anat_sentences in enumerate(anatomy_sentences):
            # sentences of the current sample i and anatomy a
            sample_anat_sentences: List[str] = _parse_sentences(sample_anat_sentences)
            all_anatomy_sentences[i].append(sample_anat_sentences)

    return all_anatomy_sentences


def _parse_sentences(sentence_list: str) -> List[str]:
    if isinstance(sentence_list, str):
        sentence_list = json.loads(sentence_list) 
    assert isinstance(sentence_list, (tuple, list))
    assert all(isinstance(sent, str) for sent in sentence_list)
    return sentence_list

# -----------> Class Labels <----------- #
def load_class_labels(data_df, config: DatasetConfig, column_prefix) -> np.ndarray:
    assert config.class_names is not None and len(config.class_names) > 0
    # one element per class, each element is a binary vector of size N
    all_class_labels: List[np.ndarray] = []
    for cls_name in config.class_names:
        assert f'{column_prefix}/{cls_name}' in data_df.columns, f'Column {column_prefix}/{cls_name} not found in data_df, available columns: {data_df.columns}'
        binary_observation = _convert_labels(
            data_df[f'{column_prefix}/{cls_name}'], uncertain_label=1 if config.uncertain_pos else 0)
        binary_observation = np.array(binary_observation, dtype=np.int64)  # (N)
        all_class_labels.append(binary_observation)
    return np.stack(all_class_labels, axis=1)  # (N x C)

def load_anatomy_class_labels(data_df, config: DatasetConfig, column_prefix) -> np.ndarray:
    assert config.class_names is not None and len(config.class_names) > 0
    assert config.anatomy_names is not None and len(config.anatomy_names) > 0
    # (N x A x C)
    anatomy_class_labels = np.zeros((len(data_df), len(config.anatomy_names), len(config.class_names)), dtype=np.int64)
    for c, cls_name in enumerate(config.class_names):
        for a, anatomy_name in enumerate(config.anatomy_names):
            col_name = f'{column_prefix}/{cls_name}/{anatomy_name}'
            if col_name not in data_df.columns:
                #log.warning(f'Anatomy label not found: {cls_name}/{anatomy_name}.'
                #             'Note thats this is expected if this combination of class and region is not present in the dataset.')
                continue
            # (N)
            binary_observation = _convert_labels(data_df[col_name], uncertain_label=1 if config.uncertain_pos else 0)
            anatomy_class_labels[:, a, c] = np.array(binary_observation, dtype=np.int64)
        if not (anatomy_class_labels[:, :, c] > 0).any():
            log.warning(f'No anatomy labels found for class {cls_name}')
    return anatomy_class_labels

def _convert_labels(labels: pd.Series, uncertain_label=1, blank_label=0):
    # convert uncertain (-1) and blank (-2) labels to the given values
    labels = labels.fillna(value=-2.0)  # -2.0 = blank
    labels = labels.to_numpy(dtype=np.int64)
    labels[labels == -1] = uncertain_label
    labels[labels == -2] = blank_label
    assert (labels >= 0).all() and (labels <= 1).all()
    return labels

# -----------> Bounding Boxes <----------- #
def load_anatomy_bboxes(data_df, config: DatasetConfig, crops, rescales, column_prefix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note: assuming that the boxes are in pixel coordinates in the (x1, y1, x2, y2) format and have not been rescaled/cropped.
    If preprocessing changing the image sizes has already been applied before dataloading, then this information is passed by crops and rescales.
    """
    assert config.anatomy_names is not None and len(config.anatomy_names) > 0
    log.info('Loading anatomical regions...')
    all_bboxes: List[np.ndarray] = []  # list of length A of arrays (N x 4), currently in pixel coordinates and (x1, y1, x2, y2) format
    all_bbox_present_masks: List[np.ndarray] = []  # list of length A of arrays (N) true if box exists
    for anat_name in tqdm.tqdm(config.anatomy_names):
        col_name = f'{column_prefix}/{anat_name}'
        # load boxes for anatomy "anat_name"
        anat_boxes: List[str] = data_df[col_name].fillna('[0, 0, 0, 0]').to_list()
        anat_boxes = [parse_bbox(box) if box is not None else None for box in anat_boxes]
        all_bboxes.append(np.array(anat_boxes, dtype=np.float32))  # (N x 4)
        # remember which anatomy boxes exist
        anat_masks = (~data_df[col_name].isna()).to_list()
        all_bbox_present_masks.append(np.array(anat_masks, dtype=np.bool8))  # (N)
    all_bboxes = np.stack(all_bboxes, axis=1)  # (N x A x 4)
    all_bbox_present_masks = np.stack(all_bbox_present_masks, axis=1) # (N x A)

    # shift, rescale, and refomat boxes
    all_bboxes[:, :, 0::2] -= crops[:, None, 0, None] / 2.
    all_bboxes[:, :, 1::2] -= crops[:, None, 1, None] / 2.
    all_bboxes[:, :, 0::2] *= rescales[:, None, 0, None]
    all_bboxes[:, :, 1::2] *= rescales[:, None, 1, None]
    all_bboxes = np.clip(all_bboxes, 0.0, 1.0)
    all_bboxes = _convert_bboxes_x1y1x2y2_to_xcyxwh(all_bboxes)

    # remove too small boxes
    box_sizes = all_bboxes[:, :, 2:4]
    empty_boxes = np.any((box_sizes <= 1e-3), axis=2)  # (N x A)
    all_bbox_present_masks[empty_boxes] = False
    
    return all_bboxes, all_bbox_present_masks

def load_anatomy_multiboxes(anatomy_bboxes, anatomy_present_masks, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    if config.multi_anatomy_names is None or len(config.multi_anatomy_names) == 0:
        return None, None
    assert len(config.anatomy_multibox_mapping) > 0
    anat_name_to_index = {anat_name: i for i, anat_name in enumerate(config.anatomy_names)}
    N = len(anatomy_bboxes)
    A_multibox = len(config.multi_anatomy_names)
    M = max(len(mapped_anatomy_names) for mapped_anatomy_names in config.anatomy_multibox_mapping.values())

    anatomy_multiboxes = np.zeros((N, A_multibox, M, 4), dtype=np.float32)
    anatomy_multibox_masks = np.zeros((N, A_multibox, M), dtype=np.bool8)
    for a, anat_name in enumerate(config.multi_anatomy_names):
        mapped_anatomy_names = config.anatomy_multibox_mapping[anat_name]
        for m, mapped_anat_name in enumerate(mapped_anatomy_names):
            mapped_anat_index = anat_name_to_index[mapped_anat_name]
            anatomy_multiboxes[:, a, m] = anatomy_bboxes[:, mapped_anat_index]
            anatomy_multibox_masks[:, a, m] = anatomy_present_masks[:, mapped_anat_index]

    return anatomy_multiboxes, anatomy_multibox_masks

def load_class_bboxes(data_df, config, crops, rescales, column_name) -> List[np.ndarray]:
    assert config.class_names is not None and len(config.class_names) > 0
    assert column_name in data_df.columns, f'Column {column_name} not found in dataframe. Available columns: {data_df.columns}'

    class_map: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(config.class_names)}

    # --- Load the observation bounding boxes, possible from several observation sets ---
    # format [[x1, y1, x2, y2, class_name]] as string
    all_bboxes: List[str] = data_df[column_name].fillna('[]').to_list()
    # format [[x1, y1, x2, y2, class_id]] as np.ndarray
    all_bboxes: List[np.ndarray] = [parse_and_filter_bboxcls_list(sampe_bboxes, class_map) for sampe_bboxes in all_bboxes]
    
    
    all_bboxes = [_convert_sample_boxes(sample_boxes, sample_crop, sample_rescales) 
                  for sample_boxes, sample_crop, sample_rescales 
                  in zip(all_bboxes, crops, rescales)]
    M_is = np.array([len(sample_boxes) for sample_boxes in all_bboxes])
    M_min, M_max = M_is.min(), M_is.max()
    M_zero = (M_is == 0).sum()
    log.info(f'Loaded {len(all_bboxes)} samples with {M_min} to {M_max} boxes per sample (zero boxes: {M_zero}/{len(all_bboxes)})')
    return all_bboxes

def _convert_sample_boxes(sample_boxes: np.ndarray, sample_crop, sample_rescales) -> np.ndarray:
        assert sample_boxes.shape[-1] >= 4, f'Expected at least 4 columns in sample_boxes, got {sample_boxes.shape}'
        # shift, rescale, and refomat boxes
        sample_boxes[..., 0:4:2] -= sample_crop[0, None] / 2.
        sample_boxes[..., 1:4:2] -= sample_crop[1, None] / 2.
        sample_boxes[..., 0:4:2] *= sample_rescales[0, None]
        sample_boxes[..., 1:4:2] *= sample_rescales[1, None]
        sample_boxes[..., :4] = np.clip(sample_boxes[..., :4], 0.0, 1.0)
        sample_boxes[..., :4] = _convert_bboxes_x1y1x2y2_to_xcyxwh(sample_boxes[..., :4])

        # remove too small boxes
        if sample_boxes.ndim == 1:
            return sample_boxes
        else:
            box_sizes = sample_boxes[..., 2:4]
            empty_boxes = np.any((box_sizes <= 1e-3), axis=-1)
            sample_boxes = sample_boxes[~empty_boxes]
            return sample_boxes

def _compute_crops_and_rescales(data_df, center_cropped=False):
    # images have already been resized during preprocessing which was not yet considered in the box coordinates
    # => rescale the box coordinates + shift them based on image cropping
    # load image sizes (one per sample)
    H, W = data_df.H.astype(float).to_numpy(),  data_df.W.astype(float).to_numpy()
    N = len(H)
    # center cropping
    crops = np.zeros((N, 2), dtype=np.float32)
    if center_cropped:
        crops[H > W, 1] = (H - W)[H > W]  # y is cropped
        crops[H < W, 0] = (W - H)[H < W]  # x is cropped
        rescales = np.zeros((N,), dtype=np.float32)
        # rescale based on the smaller dimension
        rescales[H > W] = 1. / W[H > W]
        rescales[H <= W] = 1. / H[H <= W]
        rescales = np.stack([rescales, rescales], axis=1)
    else:
        rescales = np.stack([1. / W, 1. / H], axis=1)

    return crops, rescales

def parse_bbox(bbox: Union[str, Tuple, List]) -> np.ndarray:
    """
    tuple_str: "(x1, y1, x2, y2)"
    """
    if isinstance(bbox, str):
        bbox = json.loads(bbox) # ast.literal_eval(bbox)
    assert isinstance(bbox, (tuple, list))
    assert len(bbox) == 4
    return np.array(bbox, dtype=np.float32)
    #assert tuple_str[0] == '(' and tuple_str[-1] == ')'
    #return np.array([float(val.strip()) if len(val.strip()) > 0 else float('nan') for val in tuple_str[1:-2].split(',')])

def _convert_bboxes_x1y1x2y2_to_xcyxwh(bboxes):
    size = bboxes[..., 2:4] - bboxes[..., 0:2]
    centers = bboxes[..., 0:2] + (size / 2.)
    return np.concatenate([centers, size], axis=-1)


# -----------> Images <----------- #
def preload(files: Sequence, load_fn: Callable = load_pil_gray,
            num_processes: int = min(12, os.cpu_count())) -> List:
    """
    Multiprocessing to load all files to RAM fast.

    :param files: List of file paths.
    :param load_fn: Function to load the image.
    :param num_processes: Number of processes to use.
    """
    with Pool(num_processes) as pool:
        results = pool.map(load_fn, files)
    return results


def prefetch_wrapper(load_indices: Sequence, load_fn: Callable, num_processes: int = min(12, os.cpu_count())) -> List:
    """
    Multiprocessing to load all files to RAM fast.

    :param load_fn: Function to load the image.
    :param load_indices: List of indices to load (feed to load_fn)
    :param num_processes: Number of processes to use.
    """
    from time import perf_counter
    log.info(f"Prefetching {len(load_indices)} images")
    start = perf_counter()
    images = preload(
        load_indices,
        load_fn=load_fn,
        num_processes=num_processes
    )
    log.info(f'Prefetching images took {perf_counter() - start:.2f}s')

    def new_load_fn(idx):
        return images[idx]
    new_load_indices = list(range(len(images)))
    return new_load_indices, new_load_fn


def load_image(load_indices: Sequence, load_fn: Callable, idx: int):
    return load_fn(load_indices[idx])


def parse_and_filter_bboxcls_list(bbox_list: str, class_map: Dict[str, int], has_sentence: bool = False) -> List[np.ndarray]:
    if isinstance(bbox_list, str):
        try:
            bbox_list = json.loads(bbox_list) # ast.literal_eval(bbox_list)
        except json.decoder.JSONDecodeError as e:
            log.error(f"Could not parse bbox_list: {bbox_list}")
            raise e
    assert isinstance(bbox_list, (tuple, list))
    assert all(isinstance(bbox, (tuple, list)) for bbox in bbox_list)
    assert all(len(bbox) in [5, 6] for bbox in bbox_list)
    assert all(isinstance(bbox[4], str) for bbox in bbox_list)
    assert all(len(bbox) < 6 or isinstance(bbox[5], str) for bbox in bbox_list)
    if has_sentence:
        bbox_list = [[*bbox_coords, cls_name] for *bbox_coords, cls_name, sentence in bbox_list if cls_name in class_map]
        bbox_list = np.array(bbox_list, dtype=np.float32) if len(bbox_list) > 0 else np.zeros((0, 5), dtype=np.float32)
        sentences = [sentence for *bbox_coords, cls_name, sentence in bbox_list if cls_name in class_map]
        return bbox_list, sentences
    else:
        bbox_list = [[*bbox_coords, class_map[cls_name]] for *bbox_coords, cls_name in bbox_list if cls_name in class_map]
        return np.array(bbox_list, dtype=np.float32) if len(bbox_list) > 0 else np.zeros((0, 5), dtype=np.float32)

def parse_bbox_constraint_list(bbox_list: str) -> List[List[np.ndarray]]:
    if isinstance(bbox_list, str):
        try:
            bbox_list = json.loads(bbox_list)
        except json.decoder.JSONDecodeError as e:
            log.error(f"Could not parse bbox_list: {bbox_list}")
            raise e
    bbox_list = [np.array(sent_bboxes, dtype=np.float32) for sent_bboxes in bbox_list]
    bbox_list = [sent_bboxes if sent_bboxes.size > 0 else np.zeros((0, 4), dtype=np.float32) for sent_bboxes in bbox_list]
    assert all(bbox.ndim == 2 and bbox.shape[1] == 4 for bbox in bbox_list), f'Expected 2D arrays of shape (M_i, 4), got {[bbox.shape for bbox in bbox_list]}'
    return bbox_list

def convert_bbox_list_to_padded_tensor(bboxes: List[torch.Tensor], C: int, has_class_bboxes: Optional[torch.Tensor] = None):
    if has_class_bboxes is None:
        class_ids = torch.arange(C, dtype=torch.long, device=bboxes[0].device)
    else:
        class_ids = torch.nonzero(has_class_bboxes.any(0), as_tuple=False).squeeze(-1)

    if len(class_ids) == 0 or all(len(b) == 0 for b in bboxes):
        N = len(bboxes)
        return class_ids, torch.zeros((N, C, 0, 4), dtype=torch.float32, device=class_ids.device), torch.zeros((N, C, 0,), dtype=torch.bool, device=class_ids.device)

    boxes = []
    box_masks = []
    for c in class_ids:
        boxes_c = []
        for i, b in enumerate(bboxes):
            # (M_ic, 4)
            b_c = b[b[:, -1] == c]
            boxes_c.append(b_c)
        M_c = max(len(b) for b in boxes_c)
        # (N, M_c)
        box_mask_c = torch.stack([torch.arange(M_c) < len(b) for b in boxes_c], dim=0)
        box_masks.append(box_mask_c)
        # (N, M_c, 4)
        boxes_c = torch.stack([torch.cat([b, torch.zeros((M_c - len(b), 5), dtype=b.dtype, device=b.device)], dim=0) for b in boxes_c], dim=0)
        boxes.append(boxes_c)
    M = max(b.shape[1] for b in boxes)
    # boxes: list of length C of tensors (N, M_c, 4)
    # (N, C x M)
    box_masks = torch.stack([torch.cat([b, torch.zeros((b.shape[0], M - b.shape[1]), dtype=b.dtype, device=b.device)], dim=1) for b in box_masks], dim=1)
    # (N, C x M, 4)
    boxes = torch.stack([torch.cat([b, torch.zeros((b.shape[0], M - b.shape[1], 5), dtype=b.dtype, device=b.device)], dim=1) for b in boxes], dim=1)
    
    return class_ids, boxes, box_masks
