
import os

WANDB_PROJECT = os.environ.get('WANDBPROJECT', 'chex')
WANDB_ENTITY = os.environ.get('WANDBENTITY')
MODELS_DIR = os.environ.get('LOG_DIR', os.path.expanduser("~/models/chex"))
THIRD_PARTY_MODELS_DIR = os.environ.get('THIRD_PARTY_MODELS', os.path.expanduser("~/models/third_party"))
SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.environ.get('DATA_DIR', os.path.expanduser("~/datasets"))
MIMIC_CXR_BASE_DIR = os.environ.get('MIMIC_CXR_BASE_DIR', os.path.join(DATA_DIR, 'MIMIC-CXR')) 
MIMIC_CXR_PROCESSED_DIR = os.environ.get('MIMIC_CXR_PROCESSED_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'mimic_cxr_processed'))
MIMIC_CXR_JPG_DIR = os.environ.get('MIMIC_CXR_JPG_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'mimic-cxr-jpg_2-0-0'))
MIMIC_CXR_REPORTS_DIR = os.environ.get('MIMIC_CXR_REPORT_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'mimic-cxr_2-0-0'))
CHEST_IMAGEGENOME_DIR = os.environ.get('CHEST_IMAGENOME_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'chest-imagenome-dataset-1.0.0'))
MS_CXR_DIR = os.environ.get('MS_CXR_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'ms-cxr_0-1'))

VINDR_CXR_DIR = os.environ.get('VINDR_CXR_DIR', os.path.join(DATA_DIR, 'vindr-cxr'))
VINDR_CXR_PROCESSED_DIR = os.environ.get('VINDR_CXR_PROCESSED_DIR', os.path.join(VINDR_CXR_DIR, 'processed'))

CXR14_DIR = os.environ.get('CXR14_DIR', os.path.join(DATA_DIR, 'CXR8')) 

PHYSIONET_USER = os.environ.get('PHYSIONET_USER')
PHYSIONET_PW = os.environ.get('PHYSIONET_PW')

RESOURCES_DIR = os.path.join(PROJECT_DIR, 'resources')
