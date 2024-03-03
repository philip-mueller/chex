from collections import defaultdict
from functools import partial
import getpass
from glob import glob
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from tkinter import FALSE
from typing import Dict, List, Optional, Set, Tuple
from urllib.request import HTTPBasicAuthHandler, HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm, build_opener, install_opener, urlopen
from zipfile import ZipFile
import zipfile
import click
import pandas as pd
from pyparsing import Opt, col
from PIL import ImageFile
import numpy as np
import albumentations as A
import cv2

from tqdm import tqdm
from dataset.dataset_utils import load_from_memmap, split_and_save
from dataset.text_processing.report_preprocessor import ReportProcessor, SentenceProcessor

from settings import CHEST_IMAGEGENOME_DIR, MIMIC_CXR_JPG_DIR, MIMIC_CXR_PROCESSED_DIR, MIMIC_CXR_REPORTS_DIR, MS_CXR_DIR, PHYSIONET_PW, PHYSIONET_USER
from util.data_utils import load_pil_gray


log = logging.getLogger(__name__)

MIMIC_CXR_TAGS = ['frontal', 'chexpert']
"""
- frontal: only include frontal views (AP, PA)
- report: also include the report text (split into sentences), also filter samples with no/short report
- chexpert: also include the chexpert labels, also filter samples without chexpert labels
"""
MIMIC_CXR_REPORT_TAGS = ['report']
CHEST_IMAGENOME_TAGS = [ 'cig_anatboxes', 'cig_anatlabels', 'cig_labels', 'cig_anatphrases', 'cig_phrase_assignment', 'cig_split', 'cig_nogoldleak', 'cig_noleak', 'cigmimic_split']
MSCXR_TAGS= ['mscxr_exclude', 'mscxr_boxes', 'mscxr_boxesonly', 'mscxr_valtotest', 'mscxr_traintoval']
"""
- mscxr_exclude: exclude samples that are in the MS-CXR dataset (from train, val, and test splits)
- mscxr_traintoval: move samples that are in the MS-CXR dataset from train to val split => train = train - MS-CXR-train, val = val + MS-CXR-train, test = test
- mscxr_valtotest: move MS-CXR samples from the val split to the test split => train = train, val = val - MS-CXR-val, test = test + MS-CXR-val
- mscxr_boxes: include the MS-CXR bounding boxes, also filter samples without MS-CXR bounding boxes, the splits are not affected
"""
ALL_TAGS = MIMIC_CXR_TAGS + MIMIC_CXR_REPORT_TAGS + CHEST_IMAGENOME_TAGS + MSCXR_TAGS

IMAGE_IDS_TO_IGNORE = {
    "0518c887-b80608ca-830de2d5-89acf0e2-bd3ec900",
    "03b2e67c-70631ff8-685825fb-6c989456-621ca64d",
    "786d69d0-08d16a2c-dd260165-682e66e9-acf7e942",
    "1d0bafd0-72c92e4c-addb1c57-40008638-b9ec8584",
    "f55a5fe2-395fc452-4e6b63d9-3341534a-ebb882d5",
    "14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179",
    "9c42d877-dfa63a03-a1f2eb8c-127c60c3-b20b7e01",
    "996fb121-fab58dd2-7521fd7e-f9f3133c-bc202556",
    "56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557",
    "93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6",
    "f57b4a53-5fecd631-2fe14e8a-f4780ee0-b8471007",
    "d496943d-153ec9a5-c6dfe4c0-4fb9e57f-675596eb",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50m",
    "422689b1-40e06ae8-d6151ff3-2780c186-6bd67271",
    "8385a8ad-ad5e02a8-8e1fa7f3-d822c648-2a41a205",
    "e180a7b6-684946d6-fe1782de-45ed1033-1a6f8a51",
    "f5f82c2f-e99a7a06-6ecc9991-072adb2f-497dae52",
    "6d54a492-7aade003-a238dc5c-019ccdd2-05661649",
    "2b5edbbf-116df0e3-d0fea755-fabd7b85-cbb19d84",
    "db9511e3-ee0359ab-489c3556-4a9b2277-c0bf0369",
    "87495016-a6efd89e-a3697ec7-89a81d53-627a2e13",
    "810a8e3b-2cf85e71-7ed0b3d3-531b6b68-24a5ca89",
    "a9f0620b-6e256cbd-a7f66357-2fe78c8a-49caac26",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50",
}

def load_mimic_cxr_datafile(
    dataset_name: str, split: str, image_size, load_memmap: bool = True, load_in_memory: bool = False
) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray]:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    max_size = max(image_size[0], image_size[1])
    if max_size <= 256:
        img_size_mode = 256
    else:
        img_size_mode = 512

    log.info(f'Loading MIMIC CXR dataset {dataset_name} ({split}) - size {img_size_mode}...')
    dataset_name = prepare_mimic_cxr_datasets(dataset_name)
    if load_memmap or load_in_memory:
        mmap_file, file_mapping = downsample_and_load_mimic_cxr_images(img_size_mode)
        if load_in_memory:
            log.info('Loading images in memory...')
            mmap_file = np.array(mmap_file)
    data_df = pd.read_csv(os.path.join(MIMIC_CXR_PROCESSED_DIR, f'{dataset_name}.{split}.csv'))
    data_df = data_df.astype({
        'subject_id': int,
        'study_id': int,
    })
    data_df = data_df.astype({
        'subject_id': str,
        'study_id': str,
        'dicom_id': str
    })
    data_df = data_df.copy()
    data_df['sample_id'] = pd.concat([data_df['subject_id'], data_df['study_id'], data_df['dicom_id']], axis=1).apply(lambda x: '/'.join(x), axis=1)
    
    sample_ids = data_df.sample_id.to_list()
    if load_memmap:
        log.info('Loading images from memmap...')
        indices = [file_mapping[sample_id] for sample_id in sample_ids]
        load_fn = partial(load_from_memmap, mmap_file=mmap_file)
    else:
        log.info('Loading images from jpg files...')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_dir = os.path.join(MIMIC_CXR_JPG_DIR, 'files')
        data_df['image_path'] = img_dir \
            + '/p' + data_df.subject_id.str.slice(stop=2) \
            + '/p' + data_df.subject_id \
            + '/s' + data_df.study_id \
            + '/' + data_df.dicom_id + '.jpg'
        indices = data_df.image_path
        load_fn = load_pil_gray
        img_size_mode = None
    return data_df, indices, load_fn, img_size_mode

def prepare_mimic_cxr_datasets(
    dataset_name: str,
    physionet_user: Optional[str]=None, 
    physionet_pw: Optional[str]=None,
    ):

    dataset_tags = dataset_name.split('-')
    assert dataset_tags[0] == 'mimic_cxr'
    dataset_tags = dataset_tags[1:]
    assert all(tag in ALL_TAGS for tag in dataset_tags), \
        f'Invalid dataset name {dataset_name}: Found invalid tags {[tag for tag in dataset_tags if tag not in ALL_TAGS]}. Available tags: {ALL_TAGS}'
    dataset_tags = [tag for tag in ALL_TAGS if tag in dataset_tags]
    dataset_name = 'mimic-cxr-' + '-'.join(dataset_tags)

    log.info(f'Note: Using MIMIC-CXR-PROCESSED folder: {MIMIC_CXR_PROCESSED_DIR}')
    os.makedirs(MIMIC_CXR_PROCESSED_DIR, exist_ok=True)
    if os.path.exists(os.path.join(MIMIC_CXR_PROCESSED_DIR, f'{dataset_name}.all.csv')):
        log.info(f'Dataset {dataset_name} found. Skipping preparation')
        return dataset_name
    log.info(f'Preparing dataset {dataset_name}...')

    if physionet_user is None:
        physionet_user = PHYSIONET_USER
    if physionet_user is not None and physionet_pw is None:
        physionet_pw = PHYSIONET_PW
    if physionet_user is not None and physionet_pw is None:
        physionet_pw = getpass.getpass(f'PhysioNet Password for {physionet_user}:')
        assert physionet_pw is not None

    # MIMIC CXR
    log.info(f'Note: Using MIMIC-CXR-JPG folder: {MIMIC_CXR_JPG_DIR}')
    mimic_cxr_meta_df = prepare_mimic_cxr(
        dataset_tags,
        MIMIC_CXR_JPG_DIR, MIMIC_CXR_PROCESSED_DIR,
        physionet_user=physionet_user, physionet_pw=physionet_pw)
    # MIMIC CXR REPORTS
    if any(tag in MIMIC_CXR_REPORT_TAGS for tag in dataset_tags):
        log.info(f'Note: Using MIMIC-CXR-REPORTS folder: {MIMIC_CXR_REPORTS_DIR}')
        mimic_cxr_meta_df = prepare_mimic_cxr_reports(
            dataset_tags,
            MIMIC_CXR_REPORTS_DIR, MIMIC_CXR_PROCESSED_DIR,
            physionet_user=physionet_user, physionet_pw=physionet_pw,
            mimic_cxr_meta_df=mimic_cxr_meta_df
        )
    # CHEST IMAGENOME
    if any(tag in CHEST_IMAGENOME_TAGS for tag in dataset_tags):
        log.info(f'Note: Using CHEST IMAGENOME folder: {CHEST_IMAGEGENOME_DIR}')
        mimic_cxr_meta_df = prepare_chest_imagenome(
            dataset_tags,
            CHEST_IMAGEGENOME_DIR, MIMIC_CXR_PROCESSED_DIR,
            physionet_user=physionet_user, physionet_pw=physionet_pw,
            mimic_cxr_meta_df=mimic_cxr_meta_df
        )
    # MS CXR
    if any(tag in MSCXR_TAGS for tag in dataset_tags):
        log.info(f'Note: Using MS CXR folder: {MS_CXR_DIR}')
        mimic_cxr_meta_df = prepare_ms_cxr(
            dataset_tags,
            MS_CXR_DIR, MIMIC_CXR_PROCESSED_DIR,
            physionet_user=physionet_user, physionet_pw=physionet_pw,
                mimic_cxr_meta_df=mimic_cxr_meta_df
            )

    split_and_save(mimic_cxr_meta_df, MIMIC_CXR_PROCESSED_DIR, dataset_name)

    return dataset_name

def prepare_ms_cxr(
    dataset_tags,
    mscxr_path, processed_dir,
    physionet_user, physionet_pw,
    mimic_cxr_meta_df: pd.DataFrame
):
    # download and extract
    if not os.path.exists(mscxr_path) or len(os.listdir(mscxr_path)) == 0:
        log.info(f'No MS CXR dataset found at {mscxr_path}')
        log.info('Downloading MS CXR dataset...')
        zip_file = os.path.join(mscxr_path, 'ms-cxr_0-1.zip')
        os.makedirs(mscxr_path, exist_ok=True)
        os.system(f'wget --user {physionet_user} --password {physionet_pw} -O {zip_file} https://physionet.org/content/ms-cxr/get-zip/0.1/')

        log.info('Extracting dataset...')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(mscxr_path)
        os.remove(zip_file)

    ms_cxr_annotations = pd.read_csv(
        os.path.join(
            mscxr_path, 
            'ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-0.1', 
            'MS_CXR_Local_Alignment_v1.0.0.csv'), 
        index_col='dicom_id')

    if 'mscxr_exclude' in dataset_tags:
        log.info('Removing MS-CXR samples from train/val splits')
        mimic_cxr_meta_df = mimic_cxr_meta_df[(mimic_cxr_meta_df.split == 'test') | (~mimic_cxr_meta_df.index.isin(ms_cxr_annotations.index))]
        log.info(f'MIMIC-CXR images after removing MS-CXR from train/val: {mimic_cxr_meta_df.shape[0]} records')
    if 'mscxr_boxes' in dataset_tags or 'mscxr_boxesonly' in dataset_tags:
        assert 'mscxr_exclude' not in dataset_tags
        
        ms_cxr_samples = []
        for dicom_id, group in ms_cxr_annotations.groupby('dicom_id'):
            sample_clsboxes = []
            sample_clsbboxphrases = []
            for sample_box_dict in group.to_dict('records'):
                x, y, w, h = sample_box_dict['x'], sample_box_dict['y'], sample_box_dict['w'], sample_box_dict['h']
                category_name = sample_box_dict['category_name']
                phrase = sample_box_dict['label_text']
                clsbox = [x, y, x + w, y + h, f'mscxr/{category_name}']
                sample_clsboxes.append(clsbox)
                sample_clsbboxphrases.append(phrase)

            ms_cxr_samples.append({  
                'dicom_id': dicom_id,
                'clsbbox': json.dumps(sample_clsboxes),
                'clsbboxphrases': json.dumps(sample_clsbboxphrases),
                'has_ms_cxr_boxes': True,
            })

        ms_cxr_bboxes = pd.DataFrame.from_records(ms_cxr_samples).set_index('dicom_id')
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(ms_cxr_bboxes, how='left')
        mimic_cxr_meta_df['has_ms_cxr_boxes'] = mimic_cxr_meta_df['has_ms_cxr_boxes'].fillna(False)

        if 'mscxr_boxesonly' in dataset_tags:
            mimic_cxr_meta_df = mimic_cxr_meta_df[mimic_cxr_meta_df.has_ms_cxr_boxes]
            log.info(f'MIMIC-CXR images after removing images without MS-CXR boxes: {mimic_cxr_meta_df.shape[0]} records')
        if 'mscxr_valtotest' in dataset_tags:
            # Move all validation samples to test
            mimic_cxr_meta_df.loc[(mimic_cxr_meta_df.split == 'validate') & mimic_cxr_meta_df.has_ms_cxr_boxes, 'split'] = 'test'
            log.info(f'MIMIC-CXR images after moving all validation samples to test: {mimic_cxr_meta_df.shape[0]} records')
        if 'mscxr_traintoval' in dataset_tags:
            assert 'mscxr_exclude' not in dataset_tags
            log.info('Moving train/val MS-CXR samples to val split')
            mimic_cxr_meta_df.loc[(mimic_cxr_meta_df.split != 'test') & (mimic_cxr_meta_df.has_ms_cxr_boxes), 'split'] = 'validate'
            log.info(f'MIMIC-CXR images after moving MS-CXR train/val MS-CXR samples to val split: {mimic_cxr_meta_df.shape[0]} records')

    return mimic_cxr_meta_df


def prepare_chest_imagenome(
    dataset_tags,
    chest_imagenome_path, processed_dir,
    physionet_user, physionet_pw,
    mimic_cxr_meta_df: pd.DataFrame
):
    # download and extract
    if not os.path.exists(chest_imagenome_path) or len(os.listdir(chest_imagenome_path)) == 0:
        log.info(f'No Chest ImaGenome dataset found at {chest_imagenome_path}')
        log.info('Downloading Chest ImaGenome dataset...')
        zip_file = os.path.join(chest_imagenome_path, 'chest-imagenome-dataset-1.0.0.zip')
        os.makedirs(chest_imagenome_path, exist_ok=True)
        os.system(f'wget --user {physionet_user} --password {physionet_pw} -O {zip_file} https://physionet.org/content/chest-imagenome/get-zip/1.0.0/')

        log.info('Extracting dataset...')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(chest_imagenome_path)
        os.remove(zip_file)
    
    # extract scene graphs
    silver_dir = os.path.join(chest_imagenome_path, 'chest-imagenome-dataset-1.0.0', 'silver_dataset')
    scene_graph_dir = os.path.join(silver_dir, 'scene_graph')
    if not os.path.exists(scene_graph_dir):
        log.info('Extracting scene graphs...')
        with zipfile.ZipFile(os.path.join(silver_dir, 'scene_graph.zip'), 'r') as zip_ref:
            zip_ref.extractall(silver_dir)

    # --------------------- convert scene graphs ------------------------------
    anat_bboxes_file = os.path.join(processed_dir, 'chest_imagenome-anat_bboxes.csv')
    anat_observations_file = os.path.join(processed_dir, 'chest_imagenome-anat_observations.csv')
    observation_file = os.path.join(processed_dir, 'chest_imagenome-observations.csv')
    sentences_file = os.path.join(processed_dir, 'chest_imagenome-sentences.csv')
    phrase_assignment_file = os.path.join(processed_dir, 'chest_imagenome-phrase_assignment.csv')
    if not os.path.exists(anat_bboxes_file) \
        or not os.path.exists(anat_observations_file) \
        or not os.path.exists(observation_file) \
        or not os.path.exists(sentences_file) \
        or not os.path.exists(phrase_assignment_file):
        sentence_processor = SentenceProcessor()
        scene_graphs = glob(f'{scene_graph_dir}/*.json')
        log.info('Processing scene graphs...')
        box_counts = defaultdict(int)
        invalid_boxes = []
        duplicate_boxes = []
        bbox_samples = []
        bbox_attribute_samples = []
        box_phrase_samples = []
        classification_samples = []
        unique_phrases_samples = []
        samples_with_less_than_five_boxes = 0
        samples_with_non_frontal_view = 0
        for sg_path in tqdm(scene_graphs):
            with open(sg_path, 'r') as sg_file:
                sg = json.load(sg_file)
            image_id = sg['image_id']
            patient_id = sg['patient_id']
            study_id = sg['study_id']
            viewpoint = sg['viewpoint']
            if viewpoint not in ('PA', 'AP'):
                log.warning(f'Found sample with viewpoint {viewpoint}, Skipping.')
                samples_with_non_frontal_view += 1
                continue
            if image_id not in mimic_cxr_meta_df.index:
                log.warning(f'Sample not found in processed MIMIC CXR dataset: {image_id}. Skipping')
                continue
            img_h, img_w = mimic_cxr_meta_df.at[image_id, 'H'], mimic_cxr_meta_df.at[image_id, 'W']

            # ---> find bboxes
            boxes = {}
            has_duplicate_box = False
            for box_data in sg['objects']:
                bbox_name = box_data['bbox_name']
                box_coords = (
                    max(0., min(float(img_w), float(box_data['original_x1']))), 
                    max(0., min(float(img_h), float(box_data['original_y1']))), 
                    max(0., min(float(img_w), float(box_data['original_x2']))), 
                    max(0., min(float(img_h), float(box_data['original_y2'])))
                )
                if box_data['width'] <= 0 or box_data['height'] <= 0 \
                    or (box_coords[2] - box_coords[0]) <= 0 \
                    or (box_coords[3] - box_coords[1]) <= 0:
                    invalid_boxes.append((patient_id, study_id, image_id, bbox_name))
                    continue

                if bbox_name in boxes:
                    duplicate_boxes.append((patient_id, study_id, image_id, bbox_name))
                    old_cords = boxes[bbox_name]
                    # union of bboxes
                    box_coords = (
                        min(old_cords[0], box_coords[0]), min(old_cords[1], box_coords[1]),
                        max(old_cords[2], box_coords[2]), max(old_cords[3], box_coords[3])
                    )
                    has_duplicate_box = True
                else:
                    box_counts[bbox_name] += 1
                boxes[bbox_name] = box_coords
            if len(boxes) < 5:
                log.warning(f'Found sample with only {len(boxes)} valid boxes (less than 5), Skipping.')
                samples_with_less_than_five_boxes += 1
                continue
            # ---> find attributes 
            def extract_attributes(attribute_tuples, category: str) -> List[str]:
                found_attributes = []
                for cat, relation, attr in attribute_tuples:
                    if cat == category and relation == 'yes':
                        found_attributes.append(attr)
                return found_attributes

            # -> key = bbox 
            anat_isabnormal = defaultdict(lambda: False)
            anat_diseases = defaultdict(set)
            anat_findings = defaultdict(set)
            anat_phrases = defaultdict(list)
            # -> key = phrase
            phrase_regions: Dict[str, Set[str]] = defaultdict(set)
            phrase_bboxes = defaultdict(set)
            phrase_classes: Dict[str, Set[str]] = defaultdict(set)
            phrases_isabnormal_dict: Dict[str, bool] = defaultdict(lambda: False)
            phrases_diseases_dict: Dict[str, Set[str]] = defaultdict(set)
            phrases_findings_dict: Dict[str, Set[str]] = defaultdict(set)

            for attribute_data in sg['attributes']:
                # extract and store information for this bbox
                bbox_name = attribute_data['bbox_name']
                attribute_tuples = [tuple(attr.split('|')) for sent_attr in attribute_data['attributes'] for attr in sent_attr]
                is_abnormal = ('nlp', 'yes', 'abnormal') in attribute_tuples
                disease_findings = extract_attributes(attribute_tuples, 'disease')
                anat_diseases[bbox_name].update(disease_findings)
                anatomical_findings = extract_attributes(attribute_tuples, 'anatomicalfinding')
                anat_findings[bbox_name].update(anatomical_findings)
                phrases = attribute_data['phrases']
                anat_phrases[bbox_name].extend(phrases)
                anat_isabnormal[bbox_name] = is_abnormal or anat_isabnormal[bbox_name]
                
                # extract and store information for the phrases
                phrases_attribute_tuples = [[tuple(attr.split('|')) for attr in sent_attr] for sent_attr in attribute_data['attributes']]
                phrases_isabnormal = [('nlp', 'yes', 'abnormal') in attr_tuples for attr_tuples in phrases_attribute_tuples]
                phrases_diseases = [extract_attributes(attr_tuples, 'disease') for attr_tuples in phrases_attribute_tuples]
                phrases_findings = [extract_attributes(attr_tuples, 'anatomicalfinding') for attr_tuples in phrases_attribute_tuples]
                for phrase_unprocessed, ph_isabnormal, ph_diseases, ph_findings in zip(phrases, phrases_isabnormal, phrases_diseases, phrases_findings):
                    phrase = sentence_processor.process_sentence(phrase_unprocessed)
                    phrases_isabnormal_dict[phrase] = ph_isabnormal or phrases_isabnormal_dict[phrase]
                    phrases_diseases_dict[phrase].update(ph_diseases)
                    phrases_findings_dict[phrase].update(ph_findings)
                    if bbox_name in boxes:
                        phrase_regions[phrase].add(bbox_name)
                        phrase_bboxes[phrase].add(boxes[bbox_name])

            all_diseases = set(diseases for bbox_diseases in anat_diseases.values() for diseases in bbox_diseases)
            all_anatomicalfindings = set(anatfind for bbox_anatfind in anat_findings.values() for anatfind in bbox_anatfind)
            any_is_abnormal = any(is_abnormal for is_abnormal in anat_isabnormal.values())

            unique_phrases = list(phrase_regions.keys())
            phrase_super_bboxes = {}
            phrase_classes = {}
            for phrase in unique_phrases:
                ph_bboxes = np.array(list(phrase_bboxes[phrase]))
                phrase_super_bboxes[phrase] = fuse_bboxes(ph_bboxes).tolist()

                ph_classes = []
                if phrases_isabnormal_dict[phrase]:
                    ph_classes.append('abnormal')
                else:
                    ph_classes.append('normal')
                ph_classes.extend(f'disease/{disease}' for disease in phrases_diseases_dict[phrase])
                ph_classes.extend(f'af/{anatfind}' for anatfind in phrases_findings_dict[phrase])
                phrase_classes[phrase] = ph_classes

            bbox_samples.append({
                'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
                **{f'anatbbox/{bbox_name}': json.dumps(bbox) for bbox_name, bbox in boxes.items()},
                'anat_has_duplicate_box': has_duplicate_box, 'anat_has_bboxes': len(boxes) > 0
                })
            bbox_attribute_samples.append({
                'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
                **{f'anatcls/cig/abnormal/{bbox_name}': 1.0 if is_abnormal else 0.0 for bbox_name, is_abnormal in anat_isabnormal.items()},
                **{f'anatcls/cig/normal/{bbox_name}': 0.0 if is_abnormal else 1.0 for bbox_name, is_abnormal in anat_isabnormal.items()},
                **{f'anatcls/cig/disease/{name}/{bbox_name}': 1.0 
                    for bbox_name, diseases in anat_diseases.items() for name in list(diseases) },
                **{f'anatcls/cig/af/{name}/{bbox_name}': 1.0 for bbox_name, anatfind in anat_findings.items() for name in list(anatfind)},
                })
            anat_phrases: Dict[str, List[str]] = {
                bbox_name: sentence_processor(phrases) for bbox_name, phrases in anat_phrases.items()
            }
            box_phrase_samples.append({
               'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
               **{f'anatphrases/{bbox_name}': json.dumps(phrases) for bbox_name, phrases in anat_phrases.items()},
               })
            classification_samples.append({
                'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
                'cls/cig/abnormal': 1.0 if any_is_abnormal else 0.0,
                'cls/cig/normal': 0.0 if any_is_abnormal else 1.0,
                **{f'cls/cig/disease/{disease_name}': 1.0 for disease_name in all_diseases},
                **{f'cls/cig/af/{anatfind_name}': 1.0 for anatfind_name in all_anatomicalfindings},
            })
            unique_phrases_samples.append({
                'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
                'unique_phrases': json.dumps(unique_phrases),
                'phrase_bboxes': json.dumps([phrase_super_bboxes[phrase] for phrase in unique_phrases]),
                'phrase_regions': json.dumps([list(phrase_regions[phrase]) for phrase in unique_phrases]),
                'phrasecls': json.dumps([phrase_classes[phrase] for phrase in unique_phrases]),
            })

        log.info(f'Skipped {samples_with_non_frontal_view} samples with invalid viewpoint')
        log.info(f'Skipped {samples_with_less_than_five_boxes} samples with less than 5 valid boxes')
        log.info(f'Found {len(bbox_samples)} samples with valid bboxes')
        # Save bboxes
        pd.DataFrame(bbox_samples).to_csv(anat_bboxes_file, index=False)
        duplicate_boxes_file = os.path.join(processed_dir, 'duplicated_boxes.csv')
        pd.DataFrame(duplicate_boxes, columns=['study_id', 'subject_id', 'dicom_id', 'bbox_name'])\
            .to_csv(duplicate_boxes_file, index=False)
        invalid_boxes_file = os.path.join(processed_dir, 'invalid_boxes.csv')
        pd.DataFrame(invalid_boxes, columns=['study_id', 'subject_id', 'dicom_id', 'bbox_name'])\
            .to_csv(invalid_boxes_file, index=False)
        stats_file = os.path.join(processed_dir, 'box_stats.json')
        with open(stats_file, 'w') as f:
            box_stats = {name: float(val) / len(bbox_samples) for name, val in box_counts.items()}
            json.dump(box_stats, f, indent=2, sort_keys=True)
        log.info(f'Bboxes written to file {anat_bboxes_file}')

        # Save bbox observations
        pd.DataFrame(bbox_attribute_samples).to_csv(anat_observations_file, index=False)
        log.info(f'Bboxes observations written to {anat_observations_file}')
        pd.DataFrame(classification_samples).fillna(0.0).to_csv(observation_file, index=False)
        log.info(f'Sample observations written to {observation_file}')
        pd.DataFrame(box_phrase_samples).fillna('[]').to_csv(sentences_file, index=False)
        log.info(f'Sample sentences written to {sentences_file}')
        pd.DataFrame(unique_phrases_samples).to_csv(phrase_assignment_file, index=False)
        log.info(f'Unique phrases written to {phrase_assignment_file}')

    log.info('Merging MIMIC CXR with Chest ImaGenome')
    log.info(f'MIMIC-CXR images before merge with Chest ImaGenome: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_anatboxes' in dataset_tags:
        anat_bboxes = pd.read_csv(anat_bboxes_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(anat_bboxes, how='inner')
        # we ignore samples with duplicate boxes (in train there are 9)
        mimic_cxr_meta_df = mimic_cxr_meta_df[~mimic_cxr_meta_df.anat_has_duplicate_box]
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome bboxes: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_anatlabels' in dataset_tags:
        anat_bbox_labels = pd.read_csv(anat_observations_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(anat_bbox_labels, how='inner')
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome bbox observations: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_labels' in dataset_tags:
        observation_labels = pd.read_csv(observation_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(observation_labels, how='inner')
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome sample observations: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_anatphrases' in dataset_tags:
        box_phrase_samples = pd.read_csv(sentences_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(box_phrase_samples, how='inner')
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome sentences: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_phrase_assignment' in dataset_tags:
        phrase_assignment = pd.read_csv(phrase_assignment_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(phrase_assignment, how='inner')
        mimic_cxr_meta_df = match_sentences(mimic_cxr_meta_df, 
                                            sentence_col='report_sentences', anat_phrase_col='unique_phrases', 
                                            phrase_box_col='phrase_bboxes', phrase_region_col='phrase_regions', phrase_cls_col='phrasecls')
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome phrase assignment: {mimic_cxr_meta_df.shape[0]} records')

    splits_dir = os.path.join(silver_dir, 'splits')
    if 'cig_nogoldleak' in dataset_tags or 'cig_noleak' in dataset_tags or 'cig_split' in dataset_tags or 'cigmimic_split' in dataset_tags:
        ignore_images = pd.read_csv(os.path.join(splits_dir, 'images_to_avoid.csv'), index_col='dicom_id')
        mimic_cxr_meta_df = mimic_cxr_meta_df[~mimic_cxr_meta_df.index.isin(ignore_images.index)]
        log.info(f'Avoiding {ignore_images.shape[0]} images from Chest ImaGenome Gold Dataset, remaining: {mimic_cxr_meta_df.shape[0]} records')
    splits = []
    for split in ['train', 'valid', 'test']:
        split_images = pd.read_csv(os.path.join(splits_dir, f'{split}.csv'), usecols=['dicom_id'], index_col='dicom_id')
        split_images['split'] = split
        splits.append(split_images)
    splits = pd.concat(splits)

    if 'cig_split' in dataset_tags:
        log.info('Splitting based on Chest ImaGenome splits')
        mimic_cxr_meta_df = mimic_cxr_meta_df.rename(columns={'split': 'mimic_split'}).join(splits, how='inner')
        # remove mimic cxr test samples from train/val
        mimic_cxr_meta_df = mimic_cxr_meta_df[~((mimic_cxr_meta_df.split != 'test') & (mimic_cxr_meta_df.mimic_split == 'test'))]
        mimic_cxr_meta_df = mimic_cxr_meta_df.drop(columns=['mimic_split'])
        log.info(f'MIMIC-CXR images using Chest ImaGenome splits: {mimic_cxr_meta_df.shape[0]} records')
    elif 'cigmimic_split' in dataset_tags:
        log.info('Splitting based on MIMIC CXR and Chest ImaGenome splits (test set: in both test sets, train/val: MIMIC CXR split if not in any test set)')
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(splits.rename(columns={'split': 'cig_split'}), how='inner')
        mimic_cxr_meta_df = mimic_cxr_meta_df[~((mimic_cxr_meta_df.split != 'test') & (mimic_cxr_meta_df.cig_split == 'test'))]
        mimic_cxr_meta_df = mimic_cxr_meta_df[~((mimic_cxr_meta_df.cig_split != 'test') & (mimic_cxr_meta_df.split == 'test'))]
        mimic_cxr_meta_df = mimic_cxr_meta_df.drop(columns=['cig_split'])
        log.info(f'MIMIC-CXR images using MIMIC CXR + Chest ImaGenome splits: {mimic_cxr_meta_df.shape[0]} records')
    elif 'cig_noleak' in dataset_tags:
        log.info('Splitting based on MIMIC-CXR splits but making sure no test samples from Chest ImaGenome are used in train or val.')
        test_split_images = pd.read_csv(os.path.join(splits_dir, 'test.csv'), usecols=['dicom_id'], index_col='dicom_id')
        mimic_cxr_meta_df = mimic_cxr_meta_df[(mimic_cxr_meta_df.split == 'test') | (~mimic_cxr_meta_df.index.isin(test_split_images.index))]
        log.info(f'MIMIC-CXR images after removing Chest ImaGenome test samples from trani/val: {mimic_cxr_meta_df.shape[0]} records')
    else:
        log.info('Splitting based on MIMIC-CXR splits. Chest ImaGenome test samples may be used during training.')

    return mimic_cxr_meta_df


def prepare_mimic_cxr(
    dataset_tags,
    mimic_cxr_jpg_path, processed_dir,
    physionet_user, physionet_pw,
    check_files: bool = True) -> pd.DataFrame:
    
    if not exist_mimic_cxr_jpg_metadata_files(mimic_cxr_jpg_path):
        assert physionet_user is not None and physionet_pw is not None, 'Must provide PhysioNet username and password to download MIMIC-CXR-JPG dataset'
        log.info('Downloading MIMIC-CXR-JPG metadata...')
        zip_file = os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-jpg-2.0.0.zip')
        os.makedirs(mimic_cxr_jpg_path, exist_ok=True)
        os.system(f'wget --user {physionet_user} --password {physionet_pw} -O {zip_file} https://physionet.org/content/mimic-cxr-jpg/get-zip/2.0.0/')
        
        log.info('Extracting metadata files...')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(mimic_cxr_jpg_path)
        os.remove(zip_file)

        images_found = False
    else:
        if check_files:
            images_found = exist_mimic_cxr_jpg_metadata_files(mimic_cxr_jpg_path)
        else:
            images_found = True

    if not images_found:
        log.info('Downloading MIMIC-CXR-JPG images...')
        os.system(f'wget -r -nc -nH --cut-dirs 4 -c -np -P {mimic_cxr_jpg_path}/files --user {physionet_user} --password {physionet_pw} https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/')
        log.info('Downloading images done')
    else:
        log.info('MIMIC CXR images found. Skipping download')

    mimic_cxr_meta = pd.read_csv(
        os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-metadata.csv.gz'),
        compression='gzip',
        usecols=['dicom_id', 'Rows', 'Columns', 'ViewPosition'], index_col='dicom_id')
    mimic_cxr_meta = mimic_cxr_meta.rename(columns={'Rows': 'H', 'Columns': 'W', 'ViewPosition': 'view'})

    mimic_cxr_splits = pd.read_csv(
        os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-split.csv.gz'),
        compression='gzip', index_col='dicom_id')
    mimic_cxr_meta = mimic_cxr_meta.join(mimic_cxr_splits, how='inner')
    log.info(f'Total MIMIC-CXR images: {mimic_cxr_meta.shape[0]} records')
    if 'frontal' in dataset_tags:
        mimic_cxr_meta = mimic_cxr_meta[mimic_cxr_meta.view.isin(('PA', 'AP'))]
        log.info(f'Frontal MIMIC-CXR images: {mimic_cxr_meta.shape[0]} records')

    if 'chexpert' in dataset_tags:
        mimic_cxr_chexpert = pd.read_csv(
            os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-chexpert.csv.gz'),
            compression='gzip', index_col=['subject_id', 'study_id'])
        mimic_cxr_chexpert = mimic_cxr_chexpert.rename(columns={name: f'cls/cxp/{name}' for name in mimic_cxr_chexpert.columns})
        # inner join -> we only consider samples with chexpert labels)
        mimic_cxr_meta = mimic_cxr_meta.join(mimic_cxr_chexpert, on=['subject_id', 'study_id'], how='inner')

        log.info(f'Prepared MIMIC-CXR images with CheXpert labels: {mimic_cxr_meta.shape[0]} records')

    return mimic_cxr_meta


def prepare_mimic_cxr_reports(
    dataset_tags: List[str],
    mimic_cxr_reports_path: str,
    processed_dir: str,
    physionet_user: str,
    physionet_pw: str,
    mimic_cxr_meta_df) -> pd.DataFrame:
     # download and extract
    if not os.path.exists(mimic_cxr_reports_path) or len(os.listdir(mimic_cxr_reports_path)) == 0:
        log.info(f'No MIMIC-CXR reports found at {mimic_cxr_reports_path}')
        log.info('Downloading MIMIC-CXR reports...')
        zip_file = os.path.join(mimic_cxr_reports_path, 'mimic-cxr-reports.zip')
        os.makedirs(mimic_cxr_reports_path, exist_ok=True)
        os.system(f'wget --user {physionet_user} --password {physionet_pw} -O {zip_file} https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip?download')

        log.info('Extracting dataset...')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(mimic_cxr_reports_path)
        os.remove(zip_file)

    processed_reports_file = os.path.join(processed_dir, f'reports_processed.json')
    if not os.path.exists(processed_reports_file):
        report_processor = ReportProcessor()
        processed_reports: Dict[str, List[str]] = {}
        reports = glob(f'{mimic_cxr_reports_path}/files/p*/p*/s*.txt')
        reports = list(reports)
        assert len(reports) > 0, 'No reports found'
        log.info('Processing reports...')
        for report_path in tqdm(reports):
            with open(report_path, encoding="utf-8") as f:
                full_report_text = f.read()
            study_id = report_path.split('/')[-1].split('.')[0][1:]
            processed_report: Optional[List[str]] = report_processor(full_report_text, study=f's{study_id}')
            if processed_report is None:
                continue
            processed_reports[study_id] = processed_report
        with open(processed_reports_file, 'w') as f:
            json.dump(processed_reports, f)

    with open(processed_reports_file, 'r') as f:
        processed_reports = json.load(f)
    # convert sentence lists to strings (to be compatible with pandas)
    processed_reports = {int(study_id): json.dumps(sentence_list)
                         for study_id, sentence_list in processed_reports.items()}
    
    processed_reports_df = pd.DataFrame.from_dict(processed_reports, orient='index', columns=['report_sentences'])
    processed_reports_df.index.name = 'study_id'
    log.info(f'Processed reports: {processed_reports_df.shape[0]} studies')
    log.info(f'MIMIC-CXR samples before merge: {mimic_cxr_meta_df.study_id.nunique()} studies, {mimic_cxr_meta_df.shape[0]} records')

    # inner join -> we only consider samples with reports
    mimic_cxr_meta_df = mimic_cxr_meta_df.join(processed_reports_df, on='study_id', how='inner')
    log.info(f'Prepared MIMIC-CXR images with reports: {mimic_cxr_meta_df.study_id.nunique()} studies, {mimic_cxr_meta_df.shape[0]} records')
    
    return mimic_cxr_meta_df


def downsample_and_load_mimic_cxr_images(size_mode: int) -> Tuple[np.ndarray, Dict[str, int]]:
    downsampled_path = os.path.join(MIMIC_CXR_PROCESSED_DIR, f'downsampled_{size_mode}_frontal.memmap')
    downsampled_info_path = os.path.join(MIMIC_CXR_PROCESSED_DIR, f'downsampled_{size_mode}_frontal_mapping.csv')
    if os.path.exists(downsampled_path):
        log.info(f'Using downsampled data {downsampled_path}')
        file_mapping = pd.read_csv(downsampled_info_path, usecols=['sample_id', 'index'], index_col='sample_id')
        file_mapping: Dict[str, int] = file_mapping.to_dict(orient='dict')['index']
        n_rows = len(file_mapping)
        mmap_file = np.memmap(downsampled_path, mode='r', dtype='float32', shape=(n_rows, size_mode, size_mode))
        return mmap_file, file_mapping

    assert os.path.exists(MIMIC_CXR_JPG_DIR)
    log.info(f'Downsampling images to {size_mode} (saving to {downsampled_path})...')
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img_dir = os.path.join(MIMIC_CXR_JPG_DIR, 'files')
    mimic_cxr_meta = pd.read_csv(
        os.path.join(MIMIC_CXR_JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv.gz'),
        compression='gzip',
        usecols=['dicom_id', 'Rows', 'Columns', 'ViewPosition'], index_col='dicom_id')
    mimic_cxr_splits = pd.read_csv(
        os.path.join(MIMIC_CXR_JPG_DIR, 'mimic-cxr-2.0.0-split.csv.gz'),
        compression='gzip', index_col='dicom_id')
    mimic_cxr_meta = mimic_cxr_meta.join(mimic_cxr_splits, how='inner')
    mimic_cxr_meta = mimic_cxr_meta[mimic_cxr_meta.ViewPosition.isin(('PA', 'AP'))]
    mimic_cxr_meta = mimic_cxr_meta.reset_index(drop=False)
    mimic_cxr_meta = mimic_cxr_meta.astype({
        'subject_id': int,
        'study_id': int,
    })
    mimic_cxr_meta = mimic_cxr_meta.astype({
        'subject_id': str,
        'study_id': str,
        'dicom_id': str
    })
    mimic_cxr_meta['sample_id'] = mimic_cxr_meta.subject_id + '/' + mimic_cxr_meta.study_id + '/' + mimic_cxr_meta.dicom_id
    mimic_cxr_meta['image_path'] = img_dir \
        + '/p' + mimic_cxr_meta.subject_id.str.slice(stop=2) \
        + '/p' + mimic_cxr_meta.subject_id \
        + '/s' + mimic_cxr_meta.study_id \
        + '/' + mimic_cxr_meta.dicom_id + '.jpg'

    file_mapping = []
    n_rows = mimic_cxr_meta.shape[0]
    pad_resize_transform = A.Compose([A.SmallestMaxSize(max_size=size_mode, interpolation=cv2.INTER_AREA), A.CenterCrop(height=size_mode, width=size_mode)])
    mmap_file = np.memmap(downsampled_path, mode='w+', dtype='float32', shape=(n_rows, size_mode, size_mode))
    for i, (row_index, row) in tqdm(enumerate(mimic_cxr_meta.iterrows()), total=n_rows):
        file_mapping.append((row['sample_id'], i))
        img = load_pil_gray(row['image_path'])
        img = np.array(img, dtype=np.float32) / 255.
        img = pad_resize_transform(image=img)['image']
        assert img.shape == (size_mode, size_mode)
        mmap_file[i, :, :] = img
        mmap_file.flush()
    
    pd.DataFrame(file_mapping, columns=['sample_id', 'index']).to_csv(downsampled_info_path)

    return mmap_file, {key: value for key, value in file_mapping}


# ================================= Utility functions =================================

def exist_mimic_cxr_jpg_metadata_files(mimic_cxr_jpg_path) -> bool:
    if not os.path.exists(mimic_cxr_jpg_path) or len(os.listdir(mimic_cxr_jpg_path)) == 0:
        log.info(f'No MIMIC-CXR-JPG dataset found at {mimic_cxr_jpg_path}')
        return False
    for file in ['mimic-cxr-2.0.0-metadata.csv.gz', 'mimic-cxr-2.0.0-chexpert.csv.gz', 'mimic-cxr-2.0.0-split.csv.gz']:
        if not os.path.join(mimic_cxr_jpg_path, file):
            log.warning('Metadata file not found: ' + os.path.join(mimic_cxr_jpg_path, file))
            return False
    return True

def exist_mimic_cxr_jpg_images(mimic_cxr_jpg_path) -> bool:
    log.info('Checking MIMIC CXR JPG image files...')
    meta_data_file = os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-metadata.csv.gz')
    for i, row in tqdm(pd.read_csv(meta_data_file).iterrows()):
        p = 'p' + row['subject_id']
        p_short = p[:3]
        s = 's' + row['study_id']
        jpg_file = row['dicom_id'] + '.jpg'
        path = os.path.join(mimic_cxr_jpg_path, 'files', p_short, p, s, jpg_file)
        if not os.path.exists(path):
            log.warning('Images not complete. Image not found: ' + path)
            return False
    log.info('All MIMIC CXR JPG images found')
    return True


def fuse_bboxes(bboxes: List[np.ndarray]) -> np.ndarray:
    bboxes = np.array(bboxes)
    target_boxes = []
    for bbox in bboxes:
        for i, target_box in enumerate(target_boxes):
            if are_overlapping(bbox, target_box):
                target_boxes[i] = merge_bboxes(bbox, target_box)
                break
        else:
            target_boxes.append(bbox)
    return np.array(target_boxes)
    

def are_overlapping(bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def merge_bboxes(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    return np.array([
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3]),
    ])

def match_sentences(mimic_cxr_meta_df, sentence_col, anat_phrase_col, 
                    phrase_box_col, phrase_region_col, phrase_cls_col):
    df = mimic_cxr_meta_df[[sentence_col, anat_phrase_col, phrase_box_col, phrase_region_col, phrase_cls_col]].reset_index(drop=False)
    matched_samples = []
    avg_non_matched_total = 0.
    for row in tqdm(df.to_dict(orient='records')):
        anat_phrases = json.loads(row[anat_phrase_col])
        anat_phrase_regions = json.loads(row[phrase_region_col])
        anat_phrase_classes = json.loads(row[phrase_cls_col])
        anat_phrase_boxes = json.loads(row[phrase_box_col])
        sentences = json.loads(row[sentence_col])
        non_matched_phrases = []
        phrase_mappings: Dict[int, int] = {}
        for i_anat_phrase, phrase in enumerate(anat_phrases):
            if phrase not in sentences:
                non_matched_phrases.append(phrase)
                continue
            i_sentence = sentences.index(phrase)
            phrase_mappings[i_sentence] = i_anat_phrase
        sentence_matched_mask = [i_sentence in phrase_mappings for i_sentence in range(len(sentences))]
        sentence_regions = [anat_phrase_regions[phrase_mappings[i_sentence]] if matched else [] for i_sentence, matched in enumerate(sentence_matched_mask)]
        sentence_classes = [anat_phrase_classes[phrase_mappings[i_sentence]] if matched else [] for i_sentence, matched in enumerate(sentence_matched_mask)]
        sentence_boxes = [anat_phrase_boxes[phrase_mappings[i_sentence]] if matched else [] for i_sentence, matched in enumerate(sentence_matched_mask)]
        avg_non_matched_total += len(non_matched_phrases) / len(anat_phrases) if len(anat_phrases) > 0 else 0
        matched_samples.append({
            'dicom_id': row['dicom_id'],
            'sentence_regions': sentence_regions,
            'sentence_cls': sentence_classes,
            'sentence_bbox': sentence_boxes,
        })
    avg_non_matched_total /= len(df)
    log.info(f'Average non-matched phrases: {avg_non_matched_total}')
    matched_samples_df = pd.DataFrame(matched_samples)
    mimic_cxr_meta_df = mimic_cxr_meta_df.drop([anat_phrase_col, phrase_box_col, phrase_region_col, phrase_cls_col], axis=1)
    mimic_cxr_meta_df = mimic_cxr_meta_df.join(matched_samples_df.set_index('dicom_id'), on='dicom_id', how='inner')
    return mimic_cxr_meta_df


# ================================= Main functions =================================
@click.command()
@click.argument('dataset_name')
@click.option('-u', '--physionet_user', required=True, help='Physionet username')
@click.option('-p', '--physionet_pw', prompt=True, hide_input=True, help='Physionet password')
@click.option('-d', '--mimic_cxr_dir', default=None, help='Directory to store the MIMIC CXR datasets.')
@click.option('-s', '--size_mode', default=256, help='Size of the downsampled images')
def run_prepare_mimic_cxr_datasets(dataset_name, physionet_user, physionet_pw, mimic_cxr_dir, size_mode):
    if mimic_cxr_dir is not None:
        os.environ['MIMIC_CXR_BASE_DIR'] = mimic_cxr_dir
    prepare_mimic_cxr_datasets(dataset_name, physionet_user, physionet_pw)
    downsample_and_load_mimic_cxr_images(size_mode)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_prepare_mimic_cxr_datasets()
