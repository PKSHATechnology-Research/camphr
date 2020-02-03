"""Udify: '75 Languages, 1 Model: Parsing Universal Dependencies Universally' (https://arxiv.org/abs/1904.02099)
"""
from typing import Dict, Iterable, Optional, Tuple

import spacy
import spacy.language
from spacy.language import Language
from spacy.pipeline.pipes import Sentencizer
from spacy.tokens import Doc

from camphr.types import Pathlike

from .allennlp_base import VALIDATION, AllennlpPipe
from .utils import flatten_docs_to_sents, set_heads


def load_udify_pipes(
    punct_chars: Optional[Iterable[str]] = None,
) -> Tuple[Sentencizer, "Udify"]:
    """Load udify pipe.

    Udify requires doc is sentenced, so `spacy.Sentencizer` is also created and returned.

    Returns:
        Pipes of (sentencizer, udify)

    Examples:
        >>> nlp = spacy.blank("ja")
        >>> pipes = load_udify_pipes(["。"])
        >>> [nlp.add_pipe(pipe) for pipe in pipes]
    """
    pipe = spacy.load("en_udify").get_pipe("udify")
    if punct_chars:
        sentencizer = Sentencizer(list(punct_chars))
    else:
        sentencizer = Sentencizer()
    return sentencizer, pipe


def load_udify(
    lang: str,
    punct_chars: Optional[Iterable[str]] = None,
    sentencizer: Optional[Sentencizer] = None,
) -> Language:
    """Load nlp with udify pipeline.

    Udify requires doc is sentenced, so `spacy.Sentencizer` is added to the `nlp`.

    Returns:
        `nlp` with udify and sentencizer.

    Examples:
        >>> nlp = load_udify("ja", punct_chars=["。"])
    """
    nlp = spacy.blank(lang)
    _sentencizer, udify = load_udify_pipes(punct_chars)
    nlp.add_pipe(sentencizer or _sentencizer)
    nlp.add_pipe(udify)
    return nlp


@spacy.component(
    "udify", assigns=["token.lemma", "token.dep", "token.pos", "token.head"]
)
class Udify(AllennlpPipe):
    @staticmethod
    def import_udify():
        from allennlp.common.util import import_submodules  # type: ignore

        import_submodules("udify")

    @classmethod
    def from_archive(
        cls, archive_path: Pathlike, dataset_reader_to_load: str = VALIDATION
    ):
        cls.import_udify()
        return super().from_archive(archive_path, dataset_reader_to_load)

    def __init__(self, *args, **kwargs) -> None:
        self.import_udify()
        super().__init__(*args, **kwargs)

    def set_annotations(self, docs: Iterable[Doc], outputs: Dict):
        """Set udify's output, which is calculated in self.predict, to docs"""
        from udify.models.udify_model import OUTPUTS as UdifyOUTPUTS  # type: ignore

        for sent, output in zip(flatten_docs_to_sents(docs), outputs):
            words = output[UdifyOUTPUTS.words]
            _doc_tokens = [token.text for token in sent]
            if words != _doc_tokens:
                raise ValueError(
                    "Internal error has occured."
                    f"Input text: {sent.text}\n"
                    f"Input tokens: {_doc_tokens}\n"
                    f"Model words: {words}"
                )

            for token, dep, upos, lemma in zip(
                sent,
                output[UdifyOUTPUTS.predicted_dependencies],
                output[UdifyOUTPUTS.upos],
                output[UdifyOUTPUTS.lemmas],
            ):
                token.dep_ = dep
                token.lemma_ = lemma
                token.pos_ = upos
            sent = set_heads(sent, output[UdifyOUTPUTS.predicted_heads])
            sent.doc.is_parsed = True
