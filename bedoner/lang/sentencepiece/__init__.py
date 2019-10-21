import os
import shutil
from pathlib import Path
from typing import Optional, List

import sentencepiece as spm
from spacy.attrs import LANG
from spacy.language import Language
from spacy.tokens import Doc, Token, Span


class EXTS:
    pieces_ = "spm_pieces_"
    pieces = "spm_pieces"
    alignment = "spm_alignment"


class Tokenizer:
    SPACE_CHAR = "▁"
    SPIECE_MODEL = "spiece.model"

    @staticmethod
    def install_extensions():
        Doc.set_extension(EXTS.pieces_, default=None, force=True)
        Doc.set_extension(EXTS.pieces, default=None, force=True)
        Doc.set_extension(EXTS.alignment, default=None, force=True)
        Token.set_extension(EXTS.alignment, getter=get_token_align)
        Span.set_extension(EXTS.alignment, getter=get_span_align)

    def __init__(
        self, cls: Language, nlp: Optional[Language] = None, model_path: str = ""
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
        tokens = []
        alignment = []
        buf = []
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
        self.tokenizer.load(self.model_path)

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, model_path: str):
        self._model_path = model_path
        if model_path:
            self.load_spm_tokenizer()

    def to_disk(self, path: Path, **kwargs):
        path.mkdir(exist_ok=True)
        shutil.copy(self.model_path, path / self.SPIECE_MODEL)

    def from_disk(self, path: Path, **kwargs):
        self.model_path = str((path / self.SPIECE_MODEL).absolute())


def get_token_align(token: Token) -> List[int]:
    return token.doc._.get(EXTS.alignment)[token.i]


def get_span_align(span: Span) -> List[List[int]]:
    return [token._.get(EXTS.alignment) for token in span]


class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda _text: "sentencepiece"

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

__all__ = ["SentencePieceLang"]
