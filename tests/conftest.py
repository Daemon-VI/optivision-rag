from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from optivision.types import PageRef


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def make_page(
    width: int = 512,
    height: int = 660,
    ink_rows: int = 6,
    margin: int = 60,
) -> Image.Image:
    """A white page with a few lines of dark text near the top.

    Deliberately mostly blank — that is the property the pruner must find.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = margin
    for i in range(ink_rows):
        draw.rectangle([margin, y, width - margin - (i % 3) * 40, y + 10], fill=(20, 20, 20))
        y += 26
    return img


@pytest.fixture
def page_image() -> Image.Image:
    return make_page()


@pytest.fixture
def blank_image() -> Image.Image:
    return Image.new("RGB", (512, 660), (255, 255, 255))


@pytest.fixture
def page_ref() -> PageRef:
    return PageRef(doc_id="doc", page_no=1)
