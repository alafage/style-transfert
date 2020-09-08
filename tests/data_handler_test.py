from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T

from styletrf.utils import gram_matrix, load, tensor_to_ndarray


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


def test_tensor_to_ndarray() -> None:
    transforms = T.Compose([T.ToPILImage(), T.ToTensor()])
    original_array = np.random.randint(
        0, high=256, size=(2, 2, 3), dtype=np.uint8
    )
    transformed_array = transforms(original_array)
    output_array = tensor_to_ndarray(
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(
            transformed_array
        )
    )
    assert original_array.shape == output_array.shape
    assert np.array_equal(
        np.float32(
            np.round(transformed_array.numpy().transpose(1, 2, 0), decimals=3)
        ),
        np.float32(np.round(output_array, decimals=3)),
    )
