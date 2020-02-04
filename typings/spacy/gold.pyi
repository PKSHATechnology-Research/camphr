from typing import Dict, Iterable, List, Union

from spacy.tokens.doc import Doc

class GoldParse:
    ner: Union[Iterable[str]]
    cats: Dict[str, float]

def iob_to_biluo(tags: Iterable[str]) -> List[str]: ...
def spans_from_biluo_tags(doc: Doc, tags: Iterable[str]) -> List[str]: ...

