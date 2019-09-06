from typing import Optional, List
from collections import namedtuple
from pathlib import Path
import shutil

import MeCab
from bedoner.lang.stop_words import STOP_WORDS
from .tag_map import TAG_MAP
from spacy.attrs import LANG
from spacy.language import Language
from spacy.tokens import Doc
from spacy.compat import copy_reg
from spacy.util import DummyTokenizer


ShortUnitWord = namedtuple("ShortUnitWord", ["surface", "lemma", "pos"])


def resolve_pos(token):
    """TODO"""
    return token.pos


def detailed_tokens(tokenizer: MeCab.Tagger, text: str) -> List[ShortUnitWord]:
    node = tokenizer.parseToNode(text)
    node = node.next  # first node is beginning of sentence and empty, skip it
    words = []
    while node.posid != 0:
        surface = node.surface
        base = surface  # a default value. Updated if available later.
        parts = node.feature.split(",")
        pos = ",".join(parts[0:4])
        if len(parts) > 6:
            # this information is only available for words in the tokenizer
            # dictionary
            base = parts[6]
        words.append(ShortUnitWord(surface, base, pos))
        node = node.next
    return words


class Tokenizer(DummyTokenizer):
    USERDIC = "user.dic"
    ASSETS = "assets"

    def __init__(
        self,
        cls,
        nlp: Optional[Language] = None,
        dicdir: str = None,
        userdic: str = None,
        assets: str = None,
    ):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        self.tokenizer = self.get_mecab(dicdir=dicdir, userdic=userdic)
        self.assets = assets

    def __call__(self, text):
        dtokens = detailed_tokens(self.tokenizer, text)
        words = [x.surface for x in dtokens]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        mecab_tags = []
        for token, dtoken in zip(doc, dtokens):
            mecab_tags.append(dtoken.pos)
            token.tag_ = resolve_pos(dtoken)
            token.lemma_ = dtoken.lemma
        doc.user_data["mecab_tags"] = mecab_tags
        return doc

    def get_mecab(
        self, dicdir: Optional[str] = None, userdic: Optional[str] = None
    ) -> MeCab.Tagger:
        opt = ""
        if userdic:
            opt += f"-u {userdic}"
        if dicdir:
            opt += f"-d {dicdir}"
        self.userdic = userdic
        self.dicdir = dicdir
        tokenizer = MeCab.Tagger(opt)
        tokenizer.parseToNode("")
        return tokenizer

    def to_disk(self, path: Path, **kwargs):
        path.mkdir(exist_ok=True)
        if self.userdic:
            shutil.copy(self.userdic, path / self.USERDIC)
        if self.assets:
            shutil.copy(self.assets, path / self.ASSETS)

    def from_disk(self, path: Path, **kwargs):
        userdic = (path / self.USERDIC).absolute()
        if userdic.exists():
            self.userdic = str(userdic)
        self.tokenizer = self.get_mecab(userdic=self.userdic)

        assets = (path / self.ASSETS).absolute()
        if assets.exists():
            self.assets = assets
        return self


class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda _text: "mecab"
    stop_words = STOP_WORDS
    tag_map = TAG_MAP
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(cls, nlp=None, dicdir: str = None, userdic: str = None):
        return Tokenizer(cls, nlp, dicdir=dicdir, userdic=userdic)


class Japanese(Language):
    lang = "mecab"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)


__all__ = ["Japanese"]
