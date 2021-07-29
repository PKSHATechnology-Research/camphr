from camphr_transformers.utils import get_original_spans_approx
from typing import List, Optional, Tuple
import pytest


@pytest.mark.parametrize(
    "tokens, text, expected",
    [
        (["a", "b", "c"], "abc", [(0, 1), (1, 2), (2, 3)]),
        (["a", "_", "c"], "abc", [(0, 1), (1, 2), (2, 3)]),
        (["a", "_", "c"], "abc", [(0, 1), (1, 2), (2, 3)]),
        (["a", "_", "_", "c"], "abc", [(0, 1), (1, 2), None, (2, 3)]),
    ],
)
def test_get_original_spans_approx(
    tokens: List[str], text: str, expected: List[Optional[Tuple[int, int]]]
):
    assert expected == get_original_spans_approx(tokens, text)
