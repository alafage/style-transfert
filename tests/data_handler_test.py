from pathlib import Path

import numpy as np
import torch

from styletrf.utils import load


def test_load() -> None:
    # Path as data
    path = Path(__file__).resolve().parent / "data" / "content.jpg"
    assert isinstance(load(path, 400), torch.Tensor)
    # Numpy array as data
    array = np.random.randint(0, high=256, size=(64, 64, 3))
    assert isinstance(load(array, 32), torch.Tensor)
    # Torch Tensor as data
    tensor = torch.randint(0, high=256, size=(128, 128, 3))
    assert isinstance(load(tensor, 64), torch.Tensor)
    # Check Resize
    data = np.random.randint(0, high=256, size=(128, 64, 3))
    shape = load(data, None).shape[-2:]  # with NoneType size
    assert 128 == shape[0]
    assert 64 == shape[1]
    shape = load(data, 32).shape[-2:]  # with int size
    assert 32 == min(shape)
    assert 64 == max(shape)
    shape = load(data, (100, 100)).shape[-2:]  # with tuple size
    assert 100 == shape[0] == shape[1]
