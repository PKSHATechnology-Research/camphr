import shutil
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING, Type
from camphr.doc import DocProto, Doc

if TYPE_CHECKING:
    import sentencepiece as spm  # type: ignore

from camphr.serde import SerializationMixin


class Tokenizer(SerializationMixin):
    SPACE_CHAR = "▁"
    SPIECE_MODEL = "spiece.model"
    KEY_PIECES = "spm_pieces"
    serialization_fields = ["model_path"]

    def __init__(
        self,
        model_path: str,
    ):
        self.model_path = model_path
        self.tokenizer = self.load_spm_tokenizer()

    @classmethod
    def get_spm_pieces(cls, doc: DocProto[Any]) -> List[str]:
        return doc.user_data[cls.KEY_PIECES]

    @classmethod
    def set_spm_pieces(cls, doc: DocProto[Any], pieces: List[str]):
        doc.user_data[cls.KEY_PIECES] = pieces

    def __call__(self, text: str) -> Doc:
        pieces: List[str] = self.tokenizer.EncodeAsPieces(text)  # type: ignore
        if pieces and pieces[0] == self.SPACE_CHAR:
            _tokens = pieces[1:]
        else:
            _tokens = pieces
        tokens = [token.replace(self.SPACE_CHAR, " ") for token in _tokens]
        doc = Doc.from_words(tokens)
        self.set_spm_pieces(doc, pieces)
        return doc

    def load_spm_tokenizer(self) -> "spm.SentencePieceProcessor":
        import sentencepiece as spm  # type: ignore

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(self.model_path)  # type: ignore
        return tokenizer

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, model_path: str):
        """Set model_path automatically reload sentencepiece tokenizer"""
        self._model_path = model_path
        if model_path:
            self.load_spm_tokenizer()
