"""``PageEncoderAdapter`` refuses an id list whose length differs from the pages encoded."""

from __future__ import annotations

import pytest

from optivision.adapters import PageEncoderAdapter
from optivision.encoders import SyntheticEncoder
from optivision.types import PageRef

from .conftest import make_page


def _documents():
    return [(PageRef("a", 1), make_page()), (PageRef("b", 1), make_page(ink_rows=1))]


def test_matching_ids_are_assigned():
    adapter = PageEncoderAdapter(SyntheticEncoder(dim=16, grid=8))
    corpus = adapter.encode_documents(_documents(), ids=["x", "y"])
    assert corpus.ids == ["x", "y"]


@pytest.mark.parametrize("ids", [["only-one"], ["a", "b", "c"], []])
def test_a_wrong_number_of_ids_is_refused(ids):
    adapter = PageEncoderAdapter(SyntheticEncoder(dim=16, grid=8))
    with pytest.raises(ValueError, match="ids for 2 encoded pages"):
        adapter.encode_documents(_documents(), ids=ids)
