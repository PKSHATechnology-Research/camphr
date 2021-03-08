"""The package mecab defines Japanese spacy.Language with Mecab tokenizer."""
import shutil
from pathlib import Path
from shutil import copytree
from typing import Any, List, NamedTuple, Optional, TYPE_CHECKING, Type
from typing_extensions import Literal, Protocol

if TYPE_CHECKING:
    from MeCab import Tagger

from spacy.compat import copy_reg
from spacy.language import Language
from spacy.tokens import Doc, Token

from camphr.consts import KEY_FSTRING
from camphr.lang.stop_words import STOP_WORDS
from camphr.utils import RE_URL, SerializationMixin


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


class MecabNodeProto(Protocol):
    next: "MecabNodeProto"
    surface: str
    posid: int
    feature: str
    length: int
    rlength: int


class Tokenizer(SerializationMixin):
    USERDIC = "user.dic"  # used when saving
    ASSETS = "assets"  # used when saving
    key_fstring = KEY_FSTRING

    @classmethod
    def install_extensions(cls):
        Token.set_extension(cls.key_fstring, default=None, force=True)

    def __init__(
        self,
        cls: Type["Defaults"],
        nlp: Optional[Language] = None,
        dicdir: Optional[str] = None,
        userdic: Optional[str] = None,
        assets: Optional[str] = None,
    ):
        """

        Args:
            dicdir: Mecab dictionary path. If `None`, use system configuration (~/.mecabrc or /usr/local/etc/mecabrc).
            userdic: Mecab user dictionary path. If `None`, use system configuration (~/.mecabrc or /usr/local/etc/mecabrc).
            assets: Other assets path saved with tokenizer. e.g. userdic definition csv path
        """
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        self.tokenizer = self.get_mecab(dicdir=dicdir, userdic=userdic)
        self.dictionary_type = get_dictionary_type(self.tokenizer)
        self.assets = assets

    def __call__(self, text: str) -> Doc:
        dtokens = self.detailed_tokens(text)
        words = [x.surface for x in dtokens]
        spaces = [x.space for x in dtokens]
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.tag_ = dtoken.pos
            token.lemma_ = dtoken.lemma if dtoken.lemma != "*" else token.text
            token._.set(self.key_fstring, dtoken.fstring)

        with doc.retokenize() as retokenizer:
            for match in RE_URL.finditer(doc.text):
                span = doc.char_span(*match.span())
                if span:
                    retokenizer.merge(span)
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

    def get_mecab(self, dicdir: Optional[str] = None, userdic: Optional[str] = None):
        """Create `MeCab.Tagger` instance"""
        import MeCab

        opt = ""
        if userdic:
            opt += f"-u {userdic} "
        if dicdir:
            opt += f"-d {dicdir} "
        self.userdic = userdic
        self.dicdir = dicdir
        tokenizer = MeCab.Tagger(opt.strip())
        tokenizer.parseToNode("")  # see https://github.com/explosion/spaCy/issues/2901
        return tokenizer

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
        self.tokenizer = self.get_mecab(userdic=self.userdic)

        assets = (path / self.ASSETS).absolute()
        if assets.exists():
            self.assets = str(assets)
        return self


# for pickling. see https://spacy.io/usage/adding-languages
class Defaults(Language.Defaults):  # type: ignore
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    stop_words = STOP_WORDS
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(
        cls,
        nlp: Optional[Language] = None,
        dicdir: Optional[str] = None,
        userdic: Optional[str] = None,
        assets: Optional[str] = None,
    ):
        return Tokenizer(cls, nlp, dicdir=dicdir, userdic=userdic, assets=assets)


class Japanese(Language):
    lang = "ja_mecab"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


# avoid pickling problem (see https://github.com/explosion/spaCy/issues/3191)
def pickle_japanese(instance: Any):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)
Language.factories[Japanese.lang] = Japanese

Tokenizer.install_extensions()


# for lazy loading. see https://spacy.io/usage/adding-languages
__all__ = ["Japanese"]
