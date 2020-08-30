from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T

from styletrf.utils import gram_matrix, load, unnormalize, unscale


def test_gram_matrix() -> None:
    tensor = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]]
    ).unsqueeze(0)
    expected = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    assert gram_matrix(tensor).size() == (4, 4)
    assert torch.equal(gram_matrix(tensor), expected)


def test_load() -> None:
    # Path as data
    path = Path(__file__).resolve().parent / "data" / "content.jpg"
    assert isinstance(load(path, None), torch.Tensor)
    # Numpy array as data
    array = np.random.randint(0, high=256, size=(64, 64, 3), dtype=np.uint8)
    assert isinstance(load(array, 32), torch.Tensor)
    # Torch Tensor as data
    tensor = torch.randint(0, high=256, size=(3, 128, 128), dtype=torch.uint8)
    assert isinstance(load(tensor, 64), torch.Tensor)
    # Integer as data
    integer = 5
    with pytest.raises(TypeError):
        load(integer, 20)
    # Check Resize
    data = np.random.randint(0, high=256, size=(128, 64, 3), dtype=np.uint8)
    shape = load(data, None).shape[-2:]  # with NoneType size
    assert 128 == shape[0]
    assert 64 == shape[1]
    shape = load(data, 32).shape[-2:]  # with int size
    assert 32 == min(shape)
    assert 64 == max(shape)
    shape = load(data, (100, 100)).shape[-2:]  # with tuple size
    assert 100 == shape[0] == shape[1]
    # Check range
    output = load(data, None)
    assert -1 <= output.min()
    assert 1 >= output.max()


def test_unnormalized() -> None:
    transforms = T.Compose([T.ToPILImage(), T.ToTensor()])
    original_tensor = transforms(
        torch.randint(0, high=256, size=(5, 5, 3), dtype=torch.uint8)
    )
    output = unnormalize(
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(original_tensor)
    )
    assert output.dtype == original_tensor.dtype
    assert (
        f"{output[0, 0, 0].item():.4f}"
        == f"{original_tensor[0, 0, 0].item():.4f}"
    )


def test_unscale() -> None:
    transforms = T.Compose([T.ToPILImage(), T.ToTensor()])
    original_tensor = torch.randint(
        0, high=256, size=(5, 5, 3), dtype=torch.uint8
    )
    output = unscale(transforms(original_tensor))
    assert output[0, 0, 0].item() == original_tensor[0, 0, 0].item()
