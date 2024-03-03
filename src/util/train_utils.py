
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import dataclasses
import glob
import json
import logging
import os
import random
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Collection, TypeVar, Union
import warnings
import numpy as np
from timm.scheduler import CosineLRScheduler
from torch import optim
from omegaconf import MISSING, OmegaConf
import torch
from dataset.image_transform import TransformConfig

from dataset.datasets import DatasetConfig, build_dataloader
from metrics.bootstrapping import BootstrapMetricsWrapper
from util.model_utils import prepare_config


log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    task: Optional[str] = None
    dataset: Dict[str, DatasetConfig] = MISSING

    plot_wandb: bool = True
    plot_local: bool = False
    plot_val_samples: int = 0
    plot_val_arguments: Dict[str, Any] = field(default_factory=dict)


def get_task_dataset(config: EvalConfig) -> DatasetConfig:
    return list(config.dataset.values())[0]

@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = MISSING

    model: Any = MISSING

    continue_from_checkpoint: Optional[str] = None
    load_modules_from_checkpoint: Optional[List[str]] = None

    train_dataset: Dict[str, DatasetConfig] = field(default_factory=dict)
    val_tasks: Dict[str, Any] = field(default_factory=dict)  # Dict[str, EvalConfig]
    transform: TransformConfig = MISSING
    train: bool = True
    evaluate: bool = True
    eval_mode: str = 'val'
    eval_tasks: Dict[str, Any] = field(default_factory=dict)

    batch_size: int = MISSING
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    lr: float = MISSING
    min_lr: float = MISSING
    warmup_lr: Optional[float] = MISSING
    warmup_steps: int = MISSING
    weight_decay: float = MISSING
    accumulation_steps: int = MISSING
    grad_clip_norm: Optional[float] = MISSING
    early_sopping_patience: Optional[int] = None

    metric: Optional[str] = None
    metric_mode: str = 'max'

    val_freq: Optional[int] = None
    val_ep: Optional[int] = None
    print_freq: int = MISSING
    num_workers: int = MISSING
    prefetch: bool = MISSING
    device: str = MISSING
    debug: bool = False
    compile: bool = True
    save_components: List[str] = field(default_factory=list)
    amp: bool = True


class Evaluator:
    def __init__(self, config: EvalConfig, config_cls, bootstrap: bool = False, n_bootstrap: int = 250, results_path: Optional[str] = None):
        self.config = prepare_config(config, config_cls, log)
        self.task = config.task
        self.dataset = get_task_dataset(self.config)
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.results_path = results_path
        self.metrics = {}

    @abstractmethod
    def _predict(self, **kwargs) -> tuple:
        raise NotImplementedError
    
    @abstractmethod
    def _postprocess(self, *predictions: tuple, config) -> 'BaseModelOutput':
        raise NotImplementedError

    @abstractmethod
    def _update_metrics_with_output(self, output):
        raise NotImplementedError

    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError

    def optimize_inference(self, predictions: List[tuple]) -> EvalConfig:
        warnings.warn('optimize_inference is not implemented for this model')
        return self.config


    def eval_step(self, optimize_inference=False, **kwargs) -> Union['BaseModelOutput', tuple]:
        predictions = self._predict(**kwargs)

        if optimize_inference:
            return predictions
        
        output = self._postprocess(*predictions, config=self.config)
        self._update_metrics_with_output(output)
        return output


    def _register_metric(self, metric, metric_name: Optional[str] = None):
        if metric_name is None:
            metric_name = 'metrics'
        assert metric_name not in self.metrics
        if self.bootstrap:
            metric =  BootstrapMetricsWrapper(
                metric, 
                n_bootstrap=self.n_bootstrap,
                csv_path=f'{self.results_path}_bootstrap.csv' if self.results_path is not None else None)

        self.metrics[metric_name] = metric

    def _get_metric(self, metric_name: Optional[str] = None):
        if metric_name is None:
            metric_name = 'metrics'
        assert metric_name in self.metrics
        metric = self.metrics[metric_name]
        if isinstance(metric, BootstrapMetricsWrapper):
            metric = metric.metrics
        return metric

    def _update_metric(self, metric_name: Optional[str] = None, **kwargs):
        if metric_name is None:
            metric_name = 'metrics'
        assert metric_name in self.metrics
        self.metrics[metric_name].update(**kwargs)

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _compute_metrics(self) -> dict:
        if len(self.metrics) == 1:
            return list(self.metrics.values())[0].compute()
        else:
            return {f'{metric_name}/{key}': value for metric_name, metric in self.metrics.items() for key, value in metric.compute().items()}

    def compute_metrics(self) -> dict:
        results = self._compute_metrics()
        if self.results_path is not None:
            json_path = f'{self.results_path}.json'
            json_results = {key: float(value) for key, value in results.items()}
            with open(json_path, 'w') as f:
                json.dump(json_results, f)

        return results
    

def build_optimizer(model: 'BaseModel', config: ExperimentConfig):
    return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                       weight_decay=config.weight_decay)


def build_scheduler(optimizer, config: ExperimentConfig):
    num_steps = int(config.max_steps)
    warmup_steps = int(config.warmup_steps)

    return CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.min_lr,
        warmup_lr_init=config.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )


def build_train_dataloader(config: ExperimentConfig):
    assert len(config.train_dataset) >= 1, config.train_dataset
    train_datasets = list(config.train_dataset.values())
    print('train_datasets', config.train_dataset.keys())
    primary_dataset = train_datasets[0]
    train_dl = build_dataloader(
        mode='train',
        configs=train_datasets,
        pixel_mean=primary_dataset.pixel_mean,
        pixel_std=primary_dataset.pixel_std,
        transform=config.transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch=config.prefetch)
    return train_dl

def build_val_dataloader(config: ExperimentConfig, val_task: EvalConfig):
    primary_train_dataset = list(config.train_dataset.values())[0]
    assert len(val_task.dataset) == 1, val_task.dataset
    val_dataset = list(val_task.dataset.values())[0]
    val_dl = build_dataloader(
        mode='val',
        configs=[val_dataset],
        pixel_mean=primary_train_dataset.pixel_mean,
        pixel_std=primary_train_dataset.pixel_std,
        transform=config.transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch=config.prefetch)
    return val_dl


def build_dataloaders_for_eval(config: ExperimentConfig, eval_tasks=None, eval_mode=None, load_val=False):
    if eval_tasks is None:
        eval_tasks = config.eval_tasks
    if eval_mode is None:
        eval_mode = config.eval_mode
    assert len(config.train_dataset) >= 1
    parimary_train_dataset = list(config.train_dataset.values())[0]
    for name, task in eval_tasks.items():
        assert len(task.dataset) == 1
        eval_dataset = list(task.dataset.values())[0]
        dataloader = build_dataloader(
            mode=eval_mode,
            configs=[eval_dataset],
            pixel_mean=parimary_train_dataset.pixel_mean,
            pixel_std=parimary_train_dataset.pixel_std,
            transform=config.transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch=config.prefetch)
        dataloader_val = None
        if load_val:
            dataloader_val = build_dataloader(
                mode='val',
                configs=[eval_dataset],
                pixel_mean=parimary_train_dataset.pixel_mean,
                pixel_std=parimary_train_dataset.pixel_std,
                transform=config.transform,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                prefetch=config.prefetch)
        yield name, task, dataloader, dataloader_val


def get_best_results(results, best_results, config: ExperimentConfig):
    if best_results is None:
        return results, True
    assert config.metric_mode in ('min', 'max')
    best_value = best_results['val_metric']
    value = results['val_metric']
    if (value > best_value and config.metric_mode == 'max') or \
            (value < best_value and config.metric_mode == 'min'):
        return results, True
    else:
        return best_results, False


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True

class AvgDictMeter:
    def __init__(self):
        self.values = defaultdict(float)
        self.n = 0

    def add(self, values: dict):
        for key, value in values.items():
            if value is None:
                continue
            self.values[key] += value
        self.n += 1

    def compute(self):
        return {key: value / self.n for key, value in self.values.items()}


def save_training_checkpoint(model: 'BaseModel', optimizer, lr_scheduler, scaler, results,
                             best_results, config, step, saved_components=(), is_best=False):
    saved_components = () if saved_components is None else saved_components
    saved_states = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'amp': scaler.state_dict(),
        'step': step,
        'results': results,
        'best_results': best_results,
        'experiment_config': OmegaConf.to_container(config)
    }

    # Save the current model
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', f'checkpoint_{step:09d}.pth')
    model.save_model(checkpoint_path, **saved_states)
    for component_name in saved_components:
        os.makedirs(os.path.join('checkpoints', component_name), exist_ok=True)
        checkpoint_path = os.path.join('checkpoints', component_name, f'checkpoint_{step:09d}.pth')
        model.save_model_component(checkpoint_path, component_name=component_name, **saved_states)

    # Save as best model
    if is_best:
        checkpoint_path = os.path.join('checkpoints', 'checkpoint_best.pth')
        model.save_model(checkpoint_path, **saved_states)
        for component_name in saved_components:
            checkpoint_path = os.path.join('checkpoints', component_name, 'checkpoint_best.pth')
            model.save_model_component(checkpoint_path, component_name=component_name, **saved_states)

    # Remove the previous model
    if step > 0:
        for chkpt_path in glob.glob(os.path.join('checkpoints', f'checkpoint_*.pth')):
            if not chkpt_path.endswith(f'checkpoint_{step:09d}.pth') and not chkpt_path.endswith('checkpoint_best.pth'):
                os.remove(chkpt_path)

