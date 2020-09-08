# Style Transfer Package

[![Build Status](https://travis-ci.com/alafage/style-transfert.svg?branch=master)](https://travis-ci.com/alafage/style-transfert)
[![codecov](https://codecov.io/gh/alafage/style-transfert/branch/master/graph/badge.svg)](https://codecov.io/gh/alafage/style-transfert)

A little toolbox for style transfer in python. Our implementation is based on the paper [Image Style Transfer Using Convolutional Neural Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi0vsasxNnrAhXtyIUKHVfmBTQQFjAAegQIAhAB&url=https%3A%2F%2Fwww.cv-foundation.org%2Fopenaccess%2Fcontent_cvpr_2016%2Fpapers%2FGatys_Image_Style_Transfer_CVPR_2016_paper.pdf&usg=AOvVaw1SDWBffob3bYnGb_p8WmjZ).

## Install

You can directly download the project:

```sh
pip install git+https://github.com/alafage/style-transfert.git
```

Or clone it and then install it:

```sh
git clone https://github.com/alafage/style-transfert.git
cd style-transfert
pip install .
```

## Basic Use

```python
from styletrf import StyleTRF

# Initialize the hyperparameters.
strf = StyleTRF(
    content_weight: float = 1.0,
    style_weight: float = 1e6
)

# Load the content and the style images.
strf.fit(
    content="/path/to/your/content/image",
    style="/path/to/your/style/image",
)

# Generate the target image Torch Tensor.
strf.train(iterations=100)

# Get the target image Torch Tensor.
target_image = strf.target
```
