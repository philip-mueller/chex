from functools import partial
import json
import logging
import os
from typing import Dict, Tuple
import PIL
import click
import cv2
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from dataset.dataset_utils import load_from_memmap, split_and_save
from ensemble_boxes import weighted_boxes_fusion
import pydicom
from skimage import exposure
from tqdm import tqdm
import albumentations as A

from settings import VINDR_CXR_DIR, VINDR_CXR_PROCESSED_DIR
from util.data_utils import load_pil_gray
tqdm.pandas()

log = logging.getLogger(__name__)

VINDR_CXR_DATASET_NAME = "vindrcxr"


def load_vindr_cxr_datafile(split: str, image_size, load_memmap: bool = True, load_in_memory: bool = False
) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray]:
    max_size = max(image_size[0], image_size[1])
    if max_size <= 256:
        img_size_mode = 256
    else:
        img_size_mode = 512

    log.info(f'Loading VinDr CXR dataset ({split}) - size {img_size_mode}...')
    prepare_vindr_cxr_datasets()
    if load_memmap or load_in_memory:
        mmap_file, file_mapping = downsample_and_load_vindr_cxr_images(img_size_mode)
        if load_in_memory:
            log.info('Loading images in memory...')
            mmap_file = np.array(mmap_file)
    data_df = pd.read_csv(os.path.join(VINDR_CXR_PROCESSED_DIR, f'{VINDR_CXR_DATASET_NAME}.{split}.csv'))
    data_df = data_df.copy()
    data_df = data_df.rename(columns={'image_id': 'sample_id'})
    
    sample_ids = data_df.sample_id.to_list()
    if load_memmap:
        log.info('Loading images from memmap...')
        indices = [file_mapping[sample_id] for sample_id in sample_ids]
        load_fn = partial(load_from_memmap, mmap_file=mmap_file)
    else:
        vindr_cxr_png_path = os.path.join(VINDR_CXR_PROCESSED_DIR, 'images_png')
        data_df['image_path'] = vindr_cxr_png_path \
            + '/' + data_df.sample_id + '.png'
        indices = data_df.image_path
        load_fn = load_pil_gray
        img_size_mode = None
    return data_df, indices, load_fn, img_size_mode

def prepare_vindr_cxr_datasets(val_ratio=0.5):

    vindr_path = os.path.join(VINDR_CXR_DIR, 'vindr-cxr', '1.0.0')
    vindr_cxr_png_path = os.path.join(VINDR_CXR_PROCESSED_DIR, 'images_png')

    if os.path.exists(os.path.join(VINDR_CXR_PROCESSED_DIR, f'{VINDR_CXR_DATASET_NAME}.all.csv')):
        log.info(f'Dataset {VINDR_CXR_DATASET_NAME} found. Skipping preparation')
        return 
    
    if not os.path.exists(vindr_path) or len(os.listdir(vindr_path)) == 0:
        raise NotImplementedError(f"Please download the VINDR-CXR dataset from Physionet first and store it in the folder {vindr_path}")
    
    log.info("Loading VinDr-CXR metadata...")
    # load train
    train_clslabels = pd.read_csv(
        os.path.join(vindr_path, 'annotations', 'image_labels_train.csv'),
        index_col='image_id'
    )
    train_clslabels = train_clslabels.rename(columns={'Other diseases': 'Other disease'})
    cls_names_train = list(train_clslabels.columns)
    cls_names_train.remove('rad_id')
    # majority vote of class labels
    train_clslabels = train_clslabels.groupby('image_id')[cls_names_train].agg(pd.Series.mode)
    n_train = len(train_clslabels)
    train_clslabels['split'] = ['train'] * n_train

    # load test
    valtest_clslabels = pd.read_csv(
        os.path.join(vindr_path, 'annotations', 'image_labels_test.csv'),
        index_col='image_id'
    )
    cls_names_test = list(valtest_clslabels.columns)
    assert cls_names_train == cls_names_test, f"Class names in train and test are different {cls_names_train} vs {cls_names_test}"
    # split train / val
    valtest_clslabels = valtest_clslabels.sample(frac=1, random_state=12345)
    n_valtest = len(valtest_clslabels)
    n_val = int(n_valtest * val_ratio)
    n_test = n_valtest - n_val
    valtest_clslabels['split'] = ['validate'] * n_val + ['test'] * n_test

    # load train boxes and merge them
    log.info("Loading and merging training boxes...")
    train_boxes = pd.read_csv(
        os.path.join(vindr_path, 'annotations', 'annotations_train.csv'),
        usecols=['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'],
    )
    train_boxes = train_boxes.groupby('image_id').progress_apply(fuse_boxes)
    train_dataset = train_clslabels.join(train_boxes, how='inner')

    log.info("Loading test boxes...")
    valtest_boxes = pd.read_csv(
        os.path.join(vindr_path, 'annotations', 'annotations_test.csv'),
        usecols=['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'],
    )
    valtest_boxes = valtest_boxes.groupby('image_id').progress_apply(concat_boxes)
    valtest_dataset = valtest_clslabels.join(valtest_boxes, how='inner')
    dataset = pd.concat([train_dataset, valtest_dataset], axis=0)
    dataset = dataset.rename(columns={cls_name: f'cls/vindrcxr/{cls_name}' for cls_name in cls_names_train})

    # process images, downsample + save to memmap
    log.info("Converting DICOM images...")
    os.makedirs(vindr_cxr_png_path, exist_ok=True)
    for img_id, row in tqdm(dataset.iterrows()):
        split = row['split']
        split_dir = 'train' if split == 'train' else 'test'
        H, W = convert_image(
            os.path.join(vindr_path, split_dir, f'{img_id}.dicom'),
            os.path.join(vindr_cxr_png_path, f'{img_id}.png')
        )
        dataset.loc[img_id, 'H'] = H
        dataset.loc[img_id, 'W'] = W

    split_and_save(dataset, VINDR_CXR_PROCESSED_DIR, VINDR_CXR_DATASET_NAME)
            

def convert_image(source_path, target_path):
    if not os.path.exists(target_path):
        assert os.path.exists(source_path), f"Source path {source_path} does not exist"
        # load dicom
        try:
            dcm_data = pydicom.read_file(source_path)
        except Exception as e:
            log.error(f"Failed to load dicom {source_path}")
            raise e
        img = dcm_data.pixel_array
        if dcm_data.PhotometricInterpretation == "MONOCHROME1":
            img = np.max(img) - img
        if img.dtype != np.uint8:
            img = ((img - np.min(img)) * 1.0 / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
                    
        # histogram equalization
        img = img.astype(float) / 255.
        img = exposure.equalize_hist(img)
        img = (255 * img).astype(np.uint8)
        H, W = img.shape

        # save to png
        img = PIL.Image.fromarray(img).convert('L')
        img.save(target_path, format='PNG')
    else:
        W, H = PIL.Image.open(target_path).size

    return H, W


def concat_boxes(sample: pd.DataFrame) -> pd.Series:
    sample = sample[~sample.x_min.isna()]
    class_names = sample['class_name'].tolist()
    class_names = [f'vindrcxr/{cls_name}' for cls_name in class_names]
    boxes = sample[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    boxes = [[*box, cls_name] for box, cls_name in zip(boxes, class_names)]
    return pd.Series({'clsbbox': json.dumps(boxes)})


def fuse_boxes(sample: pd.DataFrame) -> pd.Series:
    sample = sample[~sample.x_min.isna()]

    iou_thr = 0.1  # 0.5
    skip_box_thr = 0.0001

    cls_counts: Dict[str, int] = sample['class_name'].value_counts().to_dict()
    cls_names = list(cls_counts.keys())
    cls_indices = {cls_name: index for index, cls_name in enumerate(cls_names)}

    labels_single = []
    boxes_single = []
    labels_list = []
    boxes_list = []
    scores_list = []
    weights = []
    
    for cls_name, count in cls_counts.items():       
        if count == 1:
            labels_single.append(f'vindrcxr/{cls_name}')
            boxes_single.append(sample[sample.class_name==cls_name][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())
        elif count > 1:
            cls_list = sample[sample.class_name==cls_name]['class_name'].tolist()
            cls_list = [cls_indices[cls] for cls in cls_list]
            labels_list.append(cls_list)
            boxes_list.append(sample[sample.class_name==cls_name][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy())
            scores_list.append(np.ones(len(cls_list)).tolist())
            weights.append(1)
    
    if len(boxes_list) > 0:
        max_xy = max([boxes.max() for boxes in boxes_list])
        min_xy = min([boxes.min() for boxes in boxes_list])
        boxes_list = [((boxes - min_xy) / (max_xy - min_xy)).tolist() for boxes in boxes_list]
        
        # WBF
        boxes, scores, box_labels = weighted_boxes_fusion(boxes_list,
                                                        scores_list, labels_list, weights=weights,
                                                        iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes = boxes * (max_xy - min_xy) + min_xy
        boxes = boxes.tolist()
        box_labels = box_labels.astype(int).tolist()
        box_labels = [f'vindrcxr/{cls_names[cls_index]}' for cls_index in box_labels]
    else:
        boxes = []
        box_labels = []
    boxes.extend(boxes_single)
    box_labels.extend(labels_single)

    boxes = [[*box, cls_name] for box, cls_name in zip(boxes, box_labels)]
    return pd.Series({'clsbbox': json.dumps(boxes)})



def downsample_and_load_vindr_cxr_images(size_mode: int) -> Tuple[np.ndarray, Dict[str, int]]:
    assert os.path.exists(VINDR_CXR_DIR)
    downsampled_path = os.path.join(VINDR_CXR_PROCESSED_DIR, f'downsampled_{size_mode}_frontal.memmap')
    downsampled_info_path = os.path.join(VINDR_CXR_PROCESSED_DIR, f'downsampled_{size_mode}_frontal_mapping.csv')
    if os.path.exists(downsampled_path):
        log.info(f'Using downsampled data {downsampled_path}')
        file_mapping = pd.read_csv(downsampled_info_path, usecols=['sample_id', 'index'], index_col='sample_id')
        file_mapping: Dict[str, int] = file_mapping.to_dict(orient='dict')['index']
        n_rows = len(file_mapping)
        mmap_file = np.memmap(downsampled_path, mode='r', dtype='float32', shape=(n_rows, size_mode, size_mode))
        return mmap_file, file_mapping

    log.info(f'Downsampling images to {size_mode} (saving to {downsampled_path})...')
    vindr_cxr_png_path = os.path.join(VINDR_CXR_PROCESSED_DIR, 'images_png')
    metadata = pd.read_csv(os.path.join(VINDR_CXR_PROCESSED_DIR, f'{VINDR_CXR_DATASET_NAME}.all.csv'), usecols=['image_id'])
    image_ids = metadata['image_id'].tolist()

    file_mapping = []
    n_rows = len(image_ids)
    pad_resize_transform = A.Compose([A.SmallestMaxSize(max_size=size_mode, interpolation=cv2.INTER_AREA), A.CenterCrop(height=size_mode, width=size_mode)])
    mmap_file = np.memmap(downsampled_path, mode='w+', dtype='float32', shape=(n_rows, size_mode, size_mode))
    for i, image_id in tqdm(enumerate(image_ids)):
        file_mapping.append((image_id, i))
        img = load_pil_gray(os.path.join(vindr_cxr_png_path, f'{image_id}.png'))
        img = np.array(img, dtype=np.float32) / 255.
        img = pad_resize_transform(image=img)['image']
        assert img.shape == (size_mode, size_mode)
        mmap_file[i, :, :] = img
        mmap_file.flush()
    
    pd.DataFrame(file_mapping, columns=['sample_id', 'index']).to_csv(downsampled_info_path)

    return mmap_file, {key: value for key, value in file_mapping}


# ================================= Main functions =================================
@click.command()
@click.option('-r', '--val_ratio', default=0.5, help='Ratio of validation set')
@click.option('-s', '--size_mode', default=256, help='Size of the downsampled images')
def run_prepare_vindr_cxr_dataset(val_ratio, size_mode):
    prepare_vindr_cxr_datasets(val_ratio)
    downsample_and_load_vindr_cxr_images(size_mode)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_prepare_vindr_cxr_dataset()
