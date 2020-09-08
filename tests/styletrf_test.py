from pathlib import Path

import pytest
import torch

from styletrf import StyleTRF


def test_fit() -> None:
    strf = StyleTRF()
    # Check data type
    content = Path(__file__).resolve().parent / "data" / "content.jpg"
    style = Path(__file__).resolve().parent / "data" / "style.jpg"
    target = Path(__file__).resolve().parent / "data" / "target.jpg"
    strf.fit(content, style, target=target, size=200)
    assert isinstance(strf.content, torch.Tensor)
    assert isinstance(strf.style, torch.Tensor)
    assert isinstance(strf.target, torch.Tensor)
    assert strf.content.shape[0] == strf.style.shape[0] == strf.target.shape[0]
    assert strf.content.shape[1] == strf.style.shape[1] == strf.target.shape[1]
    assert 200 == min(strf.content.shape[-2:])


def test_train() -> None:
    strf = StyleTRF()
    with pytest.raises(Exception):
        strf.train(iterations=1)
    content = Path(__file__).resolve().parent / "data" / "content.jpg"
    style = Path(__file__).resolve().parent / "data" / "style.jpg"
    strf.fit(content, style, size=200)
    with pytest.warns(UserWarning):
        strf.train(iterations=1, out_path="./out.jpg")
    with pytest.warns(UserWarning):
        strf.train(iterations=1, save_every=5)
    assert strf.content.shape[0] == strf.target.shape[0]
    assert strf.content.shape[1] == strf.target.shape[1]
