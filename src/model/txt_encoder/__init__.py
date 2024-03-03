
from dataclasses import dataclass
from typing import List

from torch import BoolTensor, Tensor

from util.data_utils import TensorDataclassMixin


@dataclass
class TextEncoderOutput(TensorDataclassMixin):
    # (N x S x d) -> already projected to model space
    sentence_features: Tensor
    # (N x S)
    sentence_mask: BoolTensor
    sentences: List[List[str]]
    flattened_sentences: List[str]
    