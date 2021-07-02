import os
from pathlib import Path
import shutil
from typing import Optional, Type

from spacy.language import Language
from spacy.tokens import Doc

try:
    import sentencepiece as spm
except ImportError:
    pkgname = __name__.split(".")[0]
    raise ImportError(
        f"'sentencepiece' is not installed. Try 'pip install {pkgname}[sentencepiece]'"
    )


class EXTS:
    """For spacy.Underscore"""

    pieces_ = "spm_pieces_"


def install_extensions():
    Doc.set_extension(EXTS.pieces_, default=None, force=True)


class Tokenizer:
    SPACE_CHAR = "▁"
    SPIECE_MODEL = "spiece.model"

    def __init__(
        self,
        cls: Type["Defaults"],
        nlp: Optional[Language] = None,
        model_path: str = "",
    ):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        self.tokenizer = spm.SentencePieceProcessor()
        self.model_path = model_path

    def __call__(self, text: str) -> Doc:
        _tokens = self.tokenizer.EncodeAsPieces(text)
        spaces = [
            True if next_token.startswith(self.SPACE_CHAR) else False
            for token, next_token in zip(_tokens, _tokens[1:])
            if token != self.SPACE_CHAR
        ] + [False]
        tokens = [
            token.lstrip(self.SPACE_CHAR)
            for token in _tokens
            if token != self.SPACE_CHAR
        ]
        doc = Doc(self.vocab, tokens, spaces)
        doc._.set(EXTS.pieces_, _tokens)
        return doc

    def load_spm_tokenizer(self):
        if os.path.isdir(self.model_path):
            self.model_path = os.path.join(self.model_path, self.SPIECE_MODEL)
        try:
            self.tokenizer.load(self.model_path)
        except OSError:
            pass

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, model_path: str):
        """Set model_path automatically reload sentencepiece tokenizer"""
        self._model_path = model_path
        if model_path:
            self.load_spm_tokenizer()

    def to_disk(self, path: Path, **kwargs):
        path.mkdir(exist_ok=True)
        if self.model_path:
            shutil.copy(self.model_path, path / self.SPIECE_MODEL)

    def from_disk(self, path: Path, **kwargs):
        self.model_path = str((path / self.SPIECE_MODEL).absolute())


class Defaults(Language.Defaults):  # type: ignore
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)

    @classmethod
    def create_tokenizer(cls, nlp=None, model_path: str = ""):
        return Tokenizer(cls, nlp, model_path=model_path)


class SentencePieceLang(Language):
    lang = "sentencepiece"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


install_extensions()
Language.factories[SentencePieceLang.lang] = SentencePieceLang

__all__ = ["SentencePieceLang"]
