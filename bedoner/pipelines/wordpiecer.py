"""Module wordpiecer defines wordpiecer for pytorch transformers."""
from typing import Iterable, List

from spacy.tokens import Doc
from spacy_transformers.pipeline.wordpiecer import TransformersWordPiecer


class WordPiecer(TransformersWordPiecer):
    name = "trf_wordpiecer"

    def update(self, docs: Iterable[Doc], *args, **kwargs) -> List[Doc]:
        """Simply forward docs. This method is called when `spacy.Language.update`."""
        return [self(doc) for doc in docs]
