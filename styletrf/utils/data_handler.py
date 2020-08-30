# coding: utf-8
# fmt: off
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# fmt: on


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """ Calculate the Gram Matrix of a given tensor.
    Gram_Matrix(T) = TT'
    """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


def load(
    data: Union[str, Path, np.ndarray, torch.Tensor],
    size: Union[None, int, Tuple[int, ...]],
) -> torch.Tensor:
    """ Return a normalized Tensor from any data that is a
        PIL Image, a Numpy array or a Torch Tensor.
        Parameters:
        -----------
        data: str, Path, ndarray, Torch Tensor
            Data to transform. If data is a Numpy Array or a Torch Tensor make
            sure the data type is 'uint8'.
        size: int, default=400
            Desired output size. If size is a tuple like (h, w), output size
            will be matched to this. If size is an int, smaller edge of the
            image will be matched to this number.
        Returns
        -------
        Torch Tensor
            Normalized data tensor.
        """
    if isinstance(data, str):
        data = Path(data)

    if isinstance(data, Path):
        data_img = Image.open(data).convert("RGB")
        if data_img is not None:
            if size is None:
                size = np.array(data_img).shape[:2]
            transforms = T.Compose(
                [
                    T.Resize(size),
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            return transforms(data_img)[:3, :, :].unsqueeze(0)
        else:
            raise Exception(
                "Something went wrong while loading the content image."
            )
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        if size is None:
            size = data.shape[:2]
        transforms = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return transforms(data)[:3, :, :].unsqueeze(0)
    else:
        raise TypeError(
            "expected string, ndarray or Torch Tensor type but got ",
            f"{type(data)} instead ('content' argument).",
        )


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """ TODO
    """
    tensor = tensor.to("cpu").clone().detach()
    tensor = tensor * torch.tensor([0.5, 0.5, 0.5]) + torch.tensor(
        [0.5, 0.5, 0.5]
    )
    return tensor


def unscale(tensor: torch.Tensor) -> torch.Tensor:
    """ TODO
    """
    tensor = tensor.to("cpu").clone().detach()
    tensor *= 255
    return tensor
