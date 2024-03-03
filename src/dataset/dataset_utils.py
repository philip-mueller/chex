
import logging


import pandas as pd
import os

log = logging.getLogger(__name__)


def split_and_save(data_df, target_dir: str, target_file_prefix: str):
    os.makedirs(target_dir, exist_ok=True)
    n_samples = data_df.shape[0]
    split_file = os.path.join(target_dir, f'{target_file_prefix}.all.csv')
    data_df.to_csv(split_file)
    log.info(f'Saved {n_samples} records to {split_file}')
    for split in ('train', 'validate', 'test'):
        split_df: pd.DataFrame = data_df[data_df.split == split].drop(columns=['split'])
        if split == 'validate':
            split = 'val'
        n_split = split_df.shape[0]
        if n_split == 0:
            log.info(f'Skipping {split} split, as it has no samples')
            continue
        split_file = os.path.join(target_dir, f'{target_file_prefix}.{split}.csv')
        split_df.to_csv(split_file)
        log.info(f'Saved {n_split} records to {split_file}')
        n_samples = n_samples - n_split
    if n_samples > 0:
        log.warning(f'{n_samples} have not been saved as they are in neither of the train, val, test splits')


def load_from_memmap(index, mmap_file):
    return mmap_file[index]
