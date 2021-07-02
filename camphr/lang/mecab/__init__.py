"""The package mecab defines Japanese spacy.Language with Mecab tokenizer."""
from camphr.doc import Doc
import shutil
from pathlib import Path
from shutil import copytree
from typing import Any, List, NamedTuple, Optional, TYPE_CHECKING
from camphr.language import LanguageProto
from typing_extensions import Literal, Protocol

if TYPE_CHECKING:
    from MeCab import Tagger

from camphr.consts import KEY_FSTRING
from camphr.lang.stop_words import STOP_WORDS


class ShortUnitWord(NamedTuple):
    surface: str
    lemma: str
    pos: str
    space: bool
    fstring: str


def get_dictionary_type(
    tagger: "Tagger",
) -> Literal["ipadic", "unidic", "neologd", "juman"]:
    filename = tagger.dictionary_info().filename  # type: ignore
    for k in ["ipadic", "unidic", "neologd", "juman"]:
        if k in filename:
            return k  # type: ignore
    raise ValueError(f"Unsupported dictionary type: {filename}")


def get_mecab():
    """Create `MeCab.Tagger` instance"""
    import MeCab  # type: ignore

    tokenizer = MeCab.Tagger()
    tokenizer.parseToNode("")  # type: ignore # see https://github.com/explosion/spaCy/issues/2901
    return tokenizer


class MecabNodeProto(Protocol):
    next: "MecabNodeProto"
    surface: str
    posid: int
    feature: str
    length: int
    rlength: int


class Tokenizer:
    USERDIC = "user.dic"  # used when saving
    ASSETS = "assets"  # used when saving
    key_fstring = KEY_FSTRING

    def __init__(self):
        self.tokenizer = get_mecab()
        self.dictionary_type = get_dictionary_type(self.tokenizer)

    def __call__(self, text: str) -> Doc:
        dtokens = self.detailed_tokens(text)
        words = [x.surface for x in dtokens]
        spaces = [x.space for x in dtokens]
        doc = Doc.from_words(words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.tag_ = dtoken.pos
            token.lemma_ = dtoken.lemma if dtoken.lemma != "*" else token.text
            token.user_data[self.key_fstring] = dtoken.fstring

        doc.is_tagged = True
        return doc

    def detailed_tokens(self, text: str) -> List[ShortUnitWord]:
        """Tokenize text with Mecab and format the outputs for further processing"""
        node: MecabNodeProto = self.tokenizer.parseToNode(text)
        node = node.next
        if self.dictionary_type == "unidic":
            lemma_idx = 10
        elif self.dictionary_type == "juman":
            lemma_idx = 4
        else:
            lemma_idx = 6
        words: List[ShortUnitWord] = []
        while node.posid != 0:
            parts: List[str] = node.feature.split(",")
            pos = ",".join(parts[0:4])
            surface = node.surface
            base = parts[lemma_idx] if len(parts) > lemma_idx else surface
            nextnode = node.next
            if nextnode.length != nextnode.rlength:
                # next node contains space, so attach it to this node.
                words.append(ShortUnitWord(surface, base, pos, True, node.feature))
            elif nextnode.surface == "\u3000":
                # next node is full space, so attatch it to this node and skip the nextnode.
                words.append(ShortUnitWord(surface, base, pos, True, node.feature))
                nextnode = nextnode.next
            else:
                words.append(ShortUnitWord(surface, base, pos, False, node.feature))
            node = nextnode
        return words

    def to_disk(self, path: Path, **kwargs: Any):
        path.mkdir(exist_ok=True)
        if self.userdic:
            shutil.copy(self.userdic, path / self.USERDIC)
        if self.assets:
            copytree(self.assets, path / self.ASSETS)

    def from_disk(self, path: Path, **kwargs: Any):
        """TODO: is userdic portable?"""
        userdic = (path / self.USERDIC).absolute()
        if userdic.exists():
            self.userdic = str(userdic)
        self.tokenizer = get_mecab()

        assets = (path / self.ASSETS).absolute()
        if assets.exists():
            self.assets = str(assets)
        return self


class Japanese(LanguageProto):
    lang = "ja_mecab"

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.pipeline = []

    def __call__(self, text: str) -> Doc:
        doc = self.make_doc(text)
        for pipe in self.pipeline:
            doc = pipe(doc)
        return doc


__all__ = ["Japanese"]
