from __future__ import annotations

import os

# The CLI tests compare what a command prints with plain text. On a CI runner (GITHUB_ACTIONS
# set) typer and rich force terminal output and put colour codes inside the words, e.g. between
# the two dashes of an option. A dumb terminal makes rich print plain text everywhere; it is set
# before anything imports the CLI and is inherited by the subprocesses the tests start.
os.environ["TERM"] = "dumb"

import numpy as np
import pytest
from PIL import Image, ImageDraw

from optivision.types import PageRef


@pytest.fixture(autouse=True)
def _forget_certify_looks():
    """certify() remembers, per process, which queries it has looked at (to warn about repeated
    looks); each test starts without that memory."""
    from optivision.api import _reset_looks

    _reset_looks()
    yield


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
