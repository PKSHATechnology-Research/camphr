"""The package mecab defines Japanese spacy.Language with Mecab tokenizer."""
import shutil
from collections import namedtuple
from pathlib import Path
from shutil import copytree
from typing import List, Optional

from bedoner.consts import KEY_FSTRING
from bedoner.lang.stop_words import STOP_WORDS
from bedoner.lang.torch_mixin import TorchLanguageMixin
from bedoner.utils import RE_URL, SerializationMixin
from spacy.compat import copy_reg
from spacy.language import Language
from spacy.tokens import Doc, Token

ShortUnitWord = namedtuple(
    "ShortUnitWord", ["surface", "lemma", "pos", "space", "fstring"]
)


class Tokenizer(SerializationMixin):
    USERDIC = "user.dic"  # used when saving
    ASSETS = "assets"  # used when saving
    key_fstring = KEY_FSTRING

    @classmethod
    def install_extensions(cls):
        Token.set_extension(cls.key_fstring, default=None, force=True)

    def __init__(
        self,
        cls: Language,
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
        self.assets = assets

    def __call__(self, text: str) -> Doc:
        dtokens = self.detailed_tokens(text)
        words = [x.surface for x in dtokens]
        spaces = [x.space for x in dtokens]
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.tag_ = dtoken.pos
            token.lemma_ = dtoken.lemma
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
        node = self.tokenizer.parseToNode(text)
        node = node.next
        words: List[ShortUnitWord] = []
        while node.posid != 0:
            surface = node.surface
            base = surface
            parts = node.feature.split(",")
            pos = ",".join(parts[0:4])
            if len(parts) > 6:
                base = parts[6]
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

    def to_disk(self, path: Path, **kwargs):
        path.mkdir(exist_ok=True)
        if self.userdic:
            shutil.copy(self.userdic, path / self.USERDIC)
        if self.assets:
            copytree(self.assets, path / self.ASSETS)

    def from_disk(self, path: Path, **kwargs):
        """TODO: is userdic portable?"""
        userdic = (path / self.USERDIC).absolute()
        if userdic.exists():
            self.userdic = str(userdic)
        self.tokenizer = self.get_mecab(userdic=self.userdic)

        assets = (path / self.ASSETS).absolute()
        if assets.exists():
            self.assets = assets
        return self


# for pickling. see https://spacy.io/usage/adding-languages
class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    stop_words = STOP_WORDS
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(
        cls, nlp=None, dicdir: str = None, userdic: str = None, assets: str = None
    ):
        return Tokenizer(cls, nlp, dicdir=dicdir, userdic=userdic, assets=assets)


class Japanese(Language):
    lang = "mecab"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


class TorchJapanese(TorchLanguageMixin, Japanese):
    lang = "torch_mecab"


# avoid pickling problem (see https://github.com/explosion/spaCy/issues/3191)
def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)
Language.factories[Japanese.lang] = Japanese
Language.factories[TorchJapanese.lang] = TorchJapanese

Tokenizer.install_extensions()


# for lazy loading. see https://spacy.io/usage/adding-languages
__all__ = ["Japanese"]
