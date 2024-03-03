import dataclasses
from typing import Any, Callable, Mapping, Sequence
from PIL import Image
import torch

def load_pil_gray(path: str) -> Image.Image:
    return Image.open(path).convert('L')


class TensorDataclassMixin:
    def __init__(self):
        super(TensorDataclassMixin, self).__init__()
        assert dataclasses.is_dataclass(self), f'{type(self)} has to be a dataclass to use TensorDataclassMixin'

    def apply(self, tensor_fn: Callable[[torch.Tensor], torch.Tensor], ignore=None, apply_to_list=False):
        if ignore is None and hasattr(self, 'IGNORE_APPLY'):
            ignore = self.IGNORE_APPLY
        def apply_to_value(value):
            if value is None:
                return None
            elif isinstance(value, torch.Tensor):
                return tensor_fn(value)
            elif isinstance(value, list):
                if apply_to_list:
                    return tensor_fn(value)
                else:
                    return [apply_to_value(el) for el in value]
            elif isinstance(value, tuple):
                return tuple(apply_to_value(el) for el in value)
            elif isinstance(value, dict):
                return {key: apply_to_value(el) for key, el in value.items()}
            elif isinstance(value, TensorDataclassMixin):
                return value.apply(tensor_fn)
            else:
                return value

        def apply_to_field(field: dataclasses.Field):
            value = getattr(self, field.name)
            if ignore is not None and field.name in ignore:
                return value
            else:
                try:
                    return apply_to_value(value)
                except Exception as e:
                    raise RuntimeError(f'Error while applying {tensor_fn} to {field.name} ({value})') from e

        return self.__class__(**{field.name: apply_to_field(field) for field in dataclasses.fields(self)})

    def to(self, device, *args, non_blocking=True, **kwargs):
        return self.apply(lambda x: x.to(device, *args, non_blocking=non_blocking, **kwargs))

    def view(self, *args):
        return self.apply(lambda x: x.view(*args))

    def detach(self):
        return self.apply(lambda x: x.detach())
    
    def unsqueeze(self, dim):
        return self.apply(lambda x: x.unsqueeze(dim))
    
    def squeeze(self, dim):
        return self.apply(lambda x: x.squeeze(dim))

    def __getitem__(self, *args):
        return self.apply(lambda x: x.__getitem__(*args), apply_to_list=True)

    def to_dict(self):
        return dataclasses.asdict(self)
        

def to_device(data: Any, device: str, non_blocking=True):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        if device == 'cpu':
            non_blocking = False
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, Mapping):
        return {key: to_device(data[key], device, non_blocking=non_blocking) for key in data}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [to_device(d, device, non_blocking=non_blocking) for d in data]
    elif isinstance(data, str):
        return data
    elif isinstance(data, TensorDataclassMixin):
        if device == 'cpu':
            non_blocking = False
        return data.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError(type(data))
