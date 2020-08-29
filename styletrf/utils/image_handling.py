# coding: utf-8
# fmt: off
from typing import Any, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# fmt: on


def image_loader(
    img_path: str, max_size: int = 400, shape: Any = None,
) -> torch.Tensor:
    """ Load in and transform an image to a normalized tensor
    """
    image = Image.open(img_path).convert("RGB")
    if image is not None:
        size: Union[int, None, Tuple[int, int]]

        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape

        transforms = T.Compose(
            [
                T.Resize(size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        image = transforms(image)[:3, :, :].unsqueeze(0)

    return image


def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.to("cpu").clone().detach()
    image_arr: np.ndarray = image.numpy().squeeze()
    image_arr = image_arr.transpose(1, 2, 0)
    image_arr = image_arr * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406)
    )
    image_arr = image_arr.clip(0, 1)

    return image_arr
