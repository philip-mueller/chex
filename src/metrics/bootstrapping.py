from copy import deepcopy
from typing import Dict, List, Optional, Union
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm


import logging

from util.data_utils import to_device

log = logging.getLogger(__name__)

class BootstrapMetricsWrapper:
    def __init__(self, metrics, n_bootstrap: int = 250, csv_path: Optional[str] = None) -> None:
        self.metrics = metrics
        self.csv_path = csv_path
        self.batches = []
        self.n_bootstrap = n_bootstrap

    def update(self, **kwargs):
        self.batches.append(kwargs)

    def reset(self):
        self.batches = []

    def compute(self):
        b, all_samples = _convert_dict_batches(self.batches)

        metrics = self._bootstrap(b, all_samples)

        metric_keys = list(metrics[0].keys())
        bootstarpped_metrics = {
            key: np.stack([metric[key] for metric in metrics])
            for key in metric_keys
        }

        if self.csv_path is not None:
            df = pd.DataFrame(bootstarpped_metrics)
            log.info(f'Saving bootstrapped metrics to {self.csv_path}')
            df.to_csv(self.csv_path, index=False)
        means = {key: values.mean() for key, values in bootstarpped_metrics.items()}
        stds = {key: values.std() for key, values in bootstarpped_metrics.items()}
        #lower = {key: np.quantile(values, 0.025) for key, values in bootstarpped_metrics.items()}
        #upper = {key: np.quantile(values, 0.975) for key, values in bootstarpped_metrics.items()}

        metrics = {
            **{f'{key}/mean': mean for key, mean in means.items()},
            #**{f'{key}/lower': lower for key, lower in lower.items()},
            #**{f'{key}/upper': upper for key, upper in upper.items()},
            **{f'{key}/std': std for key, std in stds.items()}
        }

        return metrics

    def _bootstrap(self, b, all_samples):
        rng = torch.Generator().manual_seed(2147483647)
        idx = torch.arange(b)

        bootstrapped_samples = []
        for _ in range(self.n_bootstrap):
            pred_idx = idx[torch.randint(b, size=(b,), generator=rng)]  # Sample with replacement
            samples = _select_samples(all_samples, pred_idx)
            bootstrapped_samples.append(samples)

        results = []
        for samples in tqdm(bootstrapped_samples):
            metric = deepcopy(self.metrics)
            metric.update(**samples)
            metric_boot = metric.compute()
            results.append(metric_boot)
        return results
    
    
def _concat_batches(batches):
    if torch.is_tensor(batches[0]):
        return torch.cat(batches)
    else:
        return [item for batch in batches for item in batch]
    
def _convert_dict_batches(batches: List[dict]):
    assert len(batches) > 0
    keys = list(batches[0].keys())
    all_samples = {
        key: to_device(_concat_batches([batch[key] for batch in batches]), 'cpu')
        for key in keys
    }
    b = len(all_samples[keys[0]])
    return b, all_samples

def _select_samples(all_samples: Dict[str, Union[torch.Tensor, list]], indices: torch.Tensor) -> Dict[str, Union[torch.Tensor, list]]:
    return {
        key: samples[indices] if torch.is_tensor(samples) else [samples[i.item()] for i in indices]
        for key, samples in all_samples.items()
    }