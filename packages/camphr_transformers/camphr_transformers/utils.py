import textspan
from typing import List, Optional, Tuple


def get_original_spans_approx(
    tokens: List[str], text: str
) -> List[Optional[Tuple[int, int]]]:
    spans = textspan.get_original_spans(tokens, text)
    # merge_spans
    merged = [(x[0][0], x[-1][1]) if x else None for x in spans]
    # fill span in intermediate `None`
