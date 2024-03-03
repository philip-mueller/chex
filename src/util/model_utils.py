
from dataclasses import dataclass, field
import glob
from importlib import import_module
from inspect import isclass
import logging
import os
from pathlib import Path
from pkgutil import iter_modules
import types
from typing import Any, Collection, Dict, List, Optional, Tuple, Union, Type

from wandb.apis.public import Run
import torch
from torch import FloatTensor, nn
from deepdiff import DeepDiff
from omegaconf import MISSING, DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import wandb
from settings import MODELS_DIR, WANDB_ENTITY, WANDB_PROJECT
from util.data_utils import TensorDataclassMixin

log = logging.getLogger(__name__)

@dataclass
class BaseModelOutput(TensorDataclassMixin):
    IGNORE_APPLY = ('loss', 'step_metrics')
    
    loss: Optional[FloatTensor] = None
    step_metrics: Dict[str, FloatTensor] = field(default_factory=dict)


@dataclass
class BaseModelConfig:
    MODIFYABLE_ATTRIBUTES: Tuple[str, ...] = ()

    model_module: Optional[str] = None
    model_class: str = MISSING

def update_config(source_config: BaseModelConfig, override_config):
    override_config = OmegaConf.create(override_config)
    return OmegaConf.merge(source_config, override_config)


@dataclass
class MainModelConfig(BaseModelConfig):
    d_model: int = 512

    act: Any = 'relu'
    attention_dropout: float = 0.0
    dropout: float = 0.3
    droppath_prob: float = 0.2
    layer_scale: bool = True
    layer_scale_init: float = 0.1
    n_head: int = 8


class BaseModel(nn.Module):
    CONFIG_CLS: Type = None

    def __init__(self, config: Union[BaseModelConfig, dict]) -> None:
        super().__init__()
        assert self.CONFIG_CLS is not None
        config: BaseModelConfig = prepare_config(config, self.CONFIG_CLS, log)
        if 'model_class' in config:
            assert config.model_class == self.__class__.__name__
        else:
            config.model_class = self.__class__.__name__
        module_name = self.__class__.__module__.rsplit('.', 2)[-2]
        if config.model_module is not None:
            assert module_name == config.model_module.split('.')[-1], f'{module_name} != {config.model_module}'
        else:
            config.model_module = module_name
        self.config = config

    def train_step(self, **kwargs) -> BaseModelOutput:
        return self(**kwargs, compute_loss=True)

    def inference(self, **kwargs) -> BaseModelConfig:
        return self(**kwargs, compute_loss=False, return_predictions=True)

    def plot(self, model_output: BaseModelConfig, input: dict, target_dir: str, **kwargs) -> dict:
        return {}

    def build_metrics(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def save_model(self, checkpoint_path: str, **kwargs):
        config_dict: dict = OmegaConf.to_container(self.config)
        assert isinstance(config_dict, dict)
        state_dict = self.state_dict()
        ckpt_dict = {'config_dict': config_dict, 'state_dict': state_dict, **kwargs}
        torch.save(ckpt_dict, checkpoint_path)
        log.info(f'Saved checkpoint: {checkpoint_path}')

    def save_model_component(self, checkpoint_path: str, component_name: str, **kwargs):
        assert hasattr(self, component_name), f'The model {type(self)} does not have a component {component_name}'
        model_component = getattr(self, component_name)
        assert isinstance(model_component, BaseModel), f'The model component {component_name} ({type(model_component)}) of {type(self)} is not a model (does notimplement BaseMode)'
        model_component.save_model(checkpoint_path, parent_model=str(type(self)), **kwargs)


class ModelRegistry:
    registries = {}

    def __init__(self, model_module: types.ModuleType) -> None:
        assert isinstance(model_module, types.ModuleType)
        self.model_classes = {}
        module_name = model_module.__name__.rsplit('.', 1)[-1]
        self.module_name = module_name

        def register_model_class(model_cls):
            assert issubclass(model_cls, BaseModel), model_cls
            config_cls = model_cls.CONFIG_CLS
            assert config_cls is not None, f'CONFIG_CLS parameter of {model_cls} is not set'
            assert issubclass(config_cls, BaseModelConfig), config_cls
            model_cls_name = model_cls.__name__

            assert model_cls_name not in self.model_classes, f'{model_cls_name} already registered'
            self.model_classes[model_cls_name] = model_cls
            cs = ConfigStore.instance()
            cs.store(
                name=model_cls_name, 
                group=f'model/{module_name}' if module_name != 'model' else 'model', 
                node=config_cls(model_module=module_name, model_class=model_cls_name))
            log.info(f'Registered model {module_name}/{model_cls_name}')

        log.info(f'Searching models in module {module_name}...')

        package_dir = Path(model_module.__file__).resolve().parent
        print('scanning dir', package_dir)
        for (_, class_module_name, _) in iter_modules([str(package_dir.absolute())]):
            class_module = import_module(f"{model_module.__name__}.{class_module_name}")
            for attribute_name in dir(class_module):
                attribute = getattr(class_module, attribute_name)
                if isclass(attribute) and issubclass(attribute, BaseModel) and attribute is not BaseModel:
                    model_cls = attribute
                    register_model_class(model_cls)
        
        log.info(f'Registered {len(self.model_classes)} models in module {module_name}')

    def get_model_class(self, model_cls_name) -> BaseModel:
        if model_cls_name not in self.model_classes:
            raise ValueError(f'{model_cls_name} not found in module {self.module_name}.'
                             f'Available: {list(self.model_classes.keys())}')
        return self.model_classes[model_cls_name]

    @staticmethod
    def init_registries(modules: List[types.ModuleType]):
        for mod in modules:
            ModelRegistry.registry(mod)

    @staticmethod
    def registry(model_module: types.ModuleType):
        mod_name = model_module.__name__.rsplit('.', 1)[-1]

        if mod_name in ModelRegistry.registries:
            return ModelRegistry.registries[mod_name]
        else:
            registry = ModelRegistry(model_module)
            ModelRegistry.registries[mod_name] = registry
            return registry

def instantiate_model(
    config: Union[BaseModelConfig, dict], 
    config_override: Union[BaseModelConfig, dict, None] = None,
    model_module: Union[None, types.ModuleType, str] = None,
    **kwargs) -> BaseModel:

    config = OmegaConf.create(config)
    assert 'model_class' in config.keys() and 'model_module' in config.keys(), f'model_class or model_module not in config. Keys: {config.keys()}'

    if model_module is None:
        assert config.model_module is not None, f'{config}'
        model_module = config.model_module
    if isinstance(model_module, str):
        model_module = f'model.{model_module}' if model_module != 'model' else model_module
        model_module = import_module(model_module)

    registry = ModelRegistry.registry(model_module)
    model_class: Type = registry.get_model_class(config.model_class)
    assert issubclass(model_class, BaseModel), f'{model_class} of type {type(model_class)}'
    assert model_class.CONFIG_CLS is not None, f'Specify a CONFIG_CLS for the class {model_class}'

    config = prepare_config(config, model_class.CONFIG_CLS, log)
    if config_override is not None:
        config = update_config(config, config_override)
    model = model_class(config, **kwargs)
    assert isinstance(model, model_class)

    return model


def load_model_by_name(model_name: str, model_component: Optional[str] = None, step: Optional[int] = -1, 
    load_best=False, config_override: Union[BaseModelConfig, dict, None] = None, run_name=None, return_dict=False):
    model_dir = get_model_dir(model_name)
    run_dir = get_run_dir(model_dir, run_name=run_name)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    if model_component is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, model_component)

    if load_best:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
    else:
        if step == -1:
            checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.endswith('.pth') and not ckpt.endswith('_best.pth')]
            checkpoint_path = max([os.path.join(checkpoint_dir, d) for d in checkpoints], key=os.path.getmtime)
            log.info(f'Latest checkpoint: {checkpoint_path}')
        else:
            assert step is not None
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step:09d}.pth')
    return load_model_from_checkpoint(checkpoint_path, config_override=config_override, return_dict=return_dict)

def get_run_dir(model_dir, run_name=None):
    if run_name is None:
        return os.path.join(model_dir, sorted(os.listdir(model_dir))[-1])
    else:
        return os.path.join(model_dir, run_name)


def get_wandb_run_from_id(run_id) -> Run:
    return wandb.Api().run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")


def get_wandb_run_from_model_name(
    model_name: str,
    run_name: Optional[str] = None
) -> Run:
    model_dir = get_model_dir(model_name)
    run_dir = get_run_dir(model_dir, run_name)
    wandb_dir = os.path.join(run_dir, 'wandb')
    wandb_files = glob.glob(f'{wandb_dir}/run-*')
    if len(wandb_files) != 1:
        raise AssertionError(f'Multiple or no wandb runs found: '
                             f'{wandb_files}\nDir: {wandb_dir}')
    run_file = wandb_files[0]
    run_id = run_file[-8:]
    return get_wandb_run_from_id(run_id)


def load_model_from_checkpoint(checkpoint_path: str, config_override: Union[BaseModelConfig, dict, None] = None, return_dict=False):
    log.info(f'Loading model from checkpoint: {checkpoint_path}')
    ckpt_dict = torch.load(checkpoint_path)
    assert 'state_dict' in ckpt_dict and 'config_dict' in ckpt_dict, f'Invalid checkpoint dict. Keys: {ckpt_dict.keys()}'
    config_dict = ckpt_dict['config_dict']
    state_dict = ckpt_dict['state_dict']

    model = instantiate_model(config_dict, config_override=config_override, from_checkpoint=True)
    print(model.load_state_dict(state_dict, strict=False))

    if return_dict:
        return model, ckpt_dict
    else:
        return model


def get_model_dir(name: str) -> str:
    model_dir = os.path.join(MODELS_DIR, name)
    assert os.path.exists(model_dir), f'Model {name} does not exist: {model_dir}'
    model_dir_subfolders = [
        f.name
        for f in os.scandir(model_dir)
        if f.is_dir() and not f.name.startswith('.') and not f.name.startswith('eval_')]
    assert len(model_dir_subfolders) > 0, (f'Model folder of model {name} is '
                                           f'empty: {model_dir}')

    if any(f.startswith('run_') for f in model_dir_subfolders):
        return model_dir
    elif len(model_dir_subfolders) == 1:
        submodel = f'{name}/{model_dir_subfolders[0]}'
        log.info(f'Found single submodel {submodel}. Using this model')
        return get_model_dir(submodel)
    else:
        raise AssertionError(f'Model folder of model {name} ({model_dir})'
                             f' contains multiple submodels but no runs:'
                             f' {model_dir_subfolders}')


def prepare_config(config, config_cls, log):
    if isinstance(config, config_cls):
        return OmegaConf.create(config)

    # make it possible to init this class with different types of configs (dataclass, omegaconf, dict)
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)
    # fill defaults, which is required if "deprecated" configs are used (e.g. when loading old checkpoints)
    config_defaults = OmegaConf.structured(config_cls())
    new_config = OmegaConf.merge(config_defaults, config)
    diff = DeepDiff(config, new_config, verbose_level=2)
    if len(diff) > 0:
        log.info(f'Defaults have been added to the config: {diff}')
    return new_config

def get_activation(act: Any, **kwargs):
    if act is None:
        return nn.Identity()
    if not isinstance(act, str):
        return act

    if act == 'relu':
        return nn.ReLU(**kwargs)
    elif act == 'leaky_relu':
        return nn.LeakyReLU(**kwargs)
    elif act == 'elu':
        return nn.ELU(**kwargs)
    elif act == 'selu':
        return nn.SELU(**kwargs)
    elif act == 'gelu':
        return nn.GELU(**kwargs)
    elif act == 'tanh':
        return nn.Tanh(**kwargs)
    elif act == 'sigmoid':
        return nn.Sigmoid(**kwargs)
    elif act == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'Unknown activation: {act}')
