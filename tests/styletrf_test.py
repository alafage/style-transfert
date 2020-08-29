from pathlib import Path

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
