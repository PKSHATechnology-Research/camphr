import os
import shutil
from itertools import chain
from pathlib import Path
from typing import Callable, List, Optional, Type, Union

import sentencepiece as spm  # type: ignore
from spacy.language import Language
from spacy.tokens import Doc, Span, Token


class EXTS:
    """Literal declaration for spacy.Underscore"""

    pieces_ = "spm_pieces_"
    pieces = "spm_pieces"
    alignment = "spm_alignment"


class Tokenizer:
    SPACE_CHAR = "▁"
    SPIECE_MODEL = "spiece.model"

    @staticmethod
    def install_extensions():
        for text, attr in zip((True, False), (EXTS.pieces_, EXTS.pieces)):
            Doc.set_extension(attr, default=None, force=True)
            Span.set_extension(attr, getter=make_token_span_spiece_getter(text))
            Token.set_extension(attr, getter=make_token_span_spiece_getter(text))
        Doc.set_extension(EXTS.alignment, default=None, force=True)
        Span.set_extension(EXTS.alignment, getter=get_span_align)
        Token.set_extension(EXTS.alignment, getter=get_token_align)

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
        ]
        if _tokens:
            spaces += [False]
        tokens = []
        alignment = []
        buf: List[int] = []
        for i, token in enumerate(_tokens):
            if token != self.SPACE_CHAR:
                tokens.append(token.lstrip(self.SPACE_CHAR))
                alignment.append(buf + [i])
                buf = []
            else:
                buf = [i]

        doc = Doc(self.vocab, tokens, spaces)
        doc._.set(EXTS.alignment, alignment)
        doc._.set(EXTS.pieces_, _tokens)
        doc._.set(EXTS.pieces, self.tokenizer.encode_as_ids(text))
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


def get_token_align(token: Token) -> List[int]:
    return token.doc._.get(EXTS.alignment)[token.i]


def get_span_align(span: Span) -> List[List[int]]:
    return [token._.get(EXTS.alignment) for token in span]


def make_token_span_spiece_getter(text: bool) -> Callable:
    def getter(token: Union[Token, Span]) -> Union[List[str], List[int]]:
        align = token._.get(EXTS.alignment)
        if isinstance(token, Span):
            align = chain.from_iterable(align)
        if text:
            doc_spiece = token.doc._.get(EXTS.pieces_)
        else:
            doc_spiece = token.doc._.get(EXTS.pieces)
        return [doc_spiece[i] for i in align]

    return getter


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


Tokenizer.install_extensions()
Language.factories[SentencePieceLang.lang] = SentencePieceLang

__all__ = ["SentencePieceLang", "TorchSentencePieceLang"]
