# coding: utf-8
# fmt: off
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.models as M
from PIL import Image

from .utils import load, tensor_to_ndarray

# fmt: on


class StyleTRF:
    """ StyleTRF
    """

    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        style_weights: Dict[str, float] = {
            "conv1_1": 1.0,
            "conv2_1": 0.8,
            "conv3_1": 0.5,
            "conv4_1": 0.3,
            "conv5_1": 0.1,
        },
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # get the "features" portion of VGG19 (we will not need the
        # "classifier" portion)
        self.vgg = M.vgg19(pretrained=True).features
        # freeze all VGG parameters since we're only optimizing the
        # target image
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        # move the model to GPU, if available
        self.vgg.to(self.device)

        self.layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # content representation layer
            "28": "conv5_1",
        }

        self.style_weights = {
            "conv1_1": 1.0,
            "conv2_1": 0.8,
            "conv3_1": 0.5,
            "conv4_1": 0.3,
            "conv5_1": 0.1,
        }

        self.content_weight = content_weight  # alpha
        self.style_weight = style_weight  # beta

        self.content: torch.Tensor
        self.style: torch.Tensor
        self.target: torch.Tensor

    def fit(
        self,
        content: Union[str, np.ndarray, torch.Tensor],
        style: Union[str, np.ndarray, torch.Tensor],
        target: Union[None, str, np.ndarray] = None,
        size: Union[None, int, Tuple[int, ...]] = None,
    ) -> "StyleTRF":
        """ Load both content and style images into the StyleTRF class
        resizing the style one to the content image size. A target image
        can also be loaded.
        Parameters
        ----------
        content: str, Path, Numpy Array, Torch Tensor
            Content image path or ndarray.
        style: str, Path, Numpy Array, Torch Tensor
            Style image path or ndarray.
        target: str, Path, Numpy Array, Torch Tensor, default=None
            Target image path or ndarray.
        size: int, tuple, default=400
            Desired output size. If size is a tuple like (h, w), output size
            will be matched to this. If size is an int, smaller edge of the
            image will be matched to this number.
        Returns
        -------
        StyleTRF
            ...
        """
        # Content image loading
        self.content = load(data=content, size=size)
        shape = self.content.shape[-2:]
        # Style image loading
        self.style = load(data=style, size=shape)

        # Target image loading if applicable
        if target is not None:
            self.target = load(data=target, size=shape)

        return self

    def get_features(self, image: torch.Tensor) -> Any:
        """ TODO
        """
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x

        return features

    def gram_matrix(self, tensor: torch.Tensor):
        """ Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """

        # get the batch_size, depth, height, and width of the Tensor
        # reshape it, so we're multiplying the features for each channel
        # calculate the gram matrix
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())

        return gram

    def train(
        self,
        iterations: int = 1000,
        out_path: Optional[str] = None,
        save_every: Optional[int] = None,
    ):
        """ TODO
        Parameters
        ----------
        iterations: int
            Number of iterations.
        out_path: str, NoneType
            Path where the target image will be saved.
        save_every: int
            For saving the target image, intermittently.
        """
        if out_path is None and save_every is not None:
            msg = (
                "The target image won't be save because no 'out_path'"
                "argument has been provided."
            )
            warnings.warn(msg, UserWarning)
        elif out_path is not None and save_every is None:
            msg = (
                "The target image won't be save because no 'save_every'"
                "argument has been provided."
            )
            warnings.warn(msg, UserWarning)

        if self.content is not None and self.style is not None:
            # get content and style features only once before forming the
            # target image
            content_features = self.get_features(self.content)
            style_features = self.get_features(self.style)

            # calculate the gram matrices for each layer of our style
            # representation
            style_grams = {
                layer: self.gram_matrix(style_features[layer])
                for layer in style_features
            }

            # create a third "target" image and prep it for change
            # it is a good idea to start off with the target as a copy of
            # our *content* image then iteratively change its style
            self.target = (
                (self.content.clone().requires_grad_(True).to(self.device))
                if self.target is None
                else self.target.requires_grad_(True).to(self.device)
            )

            # optimizer
            optimizer = optim.Adam([self.target], lr=0.003)
            # decide how many iterations to update your image
            steps = iterations

            for ii in range(1, steps + 1):

                # TODO: get the features from your target image
                # Then calculate the content loss
                target_features = self.get_features(self.target)
                content_loss = torch.mean(
                    (target_features["conv4_2"] - content_features["conv4_2"])
                    ** 2
                )

                # the style loss
                # initialize the style loss to 0
                style_loss = 0
                # iterate through each style layer and add to the style loss
                for layer in self.style_weights:
                    # get the "target" style representation for the layer
                    target_feature = target_features[layer]
                    _, d, h, w = target_feature.shape

                    # Calculate the target gram matrix
                    target_gram = self.gram_matrix(target_feature)

                    # Get the "style" style representation
                    style_gram = style_grams[layer]
                    # Calculate the style loss for one layer, weighted
                    # appropriately
                    layer_style_loss = self.style_weights[layer] * torch.mean(
                        (target_gram - style_gram) ** 2
                    )

                    # add to the style loss
                    style_loss += layer_style_loss / (d * h * w)

                # Calculate the *total* loss
                total_loss = (
                    self.content_weight * content_loss
                    + self.style_weight * style_loss
                )

                # update your target image
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # display intermediate images and print the loss
                if ii % 100 == 0:
                    print(f"Iteration: {ii} - Total loss: {total_loss.item()}")
                if isinstance(save_every, int) and isinstance(out_path, str):
                    if ii % save_every == 0:
                        # plt.imshow(tensor_to_ndarray(self.target))
                        self.save_target(out_path)
                        # plt.show()
        else:
            raise Exception(
                "You need to provide both content and style images."
            )

    def save_target(self, path: str):
        """ TODO
        """
        if isinstance(self.target, torch.Tensor):
            array = tensor_to_ndarray(self.target) * 255
            img = Image.fromarray(np.unint8(array))
            img.save(path)
        else:
            raise Exception("No target image to save.")

    def content_style_plotter(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        if isinstance(self.content, torch.Tensor):
            ax1.imshow(tensor_to_ndarray(self.content))
        if isinstance(self.style, torch.Tensor):
            ax2.imshow(tensor_to_ndarray(self.style))

        plt.show()
