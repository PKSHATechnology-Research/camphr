from typing import List, Tuple, Optional, Callable, Dict
import spacy.vocab
from spacy.tokens import Doc

Matches = List[Tuple[str, int, int]]

class Matcher:
    def __init__(self, vocab: spacy.vocab.Vocab, validate: bool = False): ...
    def __call__(self, doc: Doc) -> Matches: ...
    def add(
        self,
        match_id: str,
        on_match: Optional[Callable[["Matcher", Doc, str, Matches]]],
        *patterns: List[Dict[str, str]]
    ): ...
