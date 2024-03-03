
from dataclasses import dataclass
from typing import Optional

from torch import BoolTensor, Tensor

from util.data_utils import TensorDataclassMixin


@dataclass
class ImageEncoderOutput(TensorDataclassMixin):
    # (N x H x W x d) -> already projected to model space
    patch_features: Tensor
    # (N x H x W x d)
    pos_embeddings: Tensor
    # (N x d)
    global_features: Optional[Tensor] = None

    @property
    def device(self):
        return self.patch_features.device
    
    @property
    def dtype(self):
        return self.patch_features.dtype
    
    @property
    def N(self):
        return self.patch_features.shape[0]
    
    @property
    def d(self):
        return self.patch_features.shape[-1]
    
