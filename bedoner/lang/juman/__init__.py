import re
from typing import Optional, List, Dict, Any
from collections import namedtuple

from pyknp import Juman, Morpheme
from .stop_words import STOP_WORDS
from .tag_map import TAG_MAP
from spacy.attrs import LANG
from spacy.language import Language
from spacy.tokens import Doc
from spacy.compat import copy_reg
from spacy.util import DummyTokenizer


ShortUnitWord = namedtuple("ShortUnitWord", ["surface", "lemma", "pos"])


def detailed_tokens(tokenizer: Juman, text) -> List[Morpheme]:
    """Format Mecab output into a nice data structure, based on Janome."""
    words = []
    ml = tokenizer.analysis(text)
    for m in ml:
        m: Morpheme = m
        surface = m.midasi
        pos = m.hinsi + "/" + m.bunrui
        lemma = m.genkei or surface
        words.append(ShortUnitWord(surface, lemma, pos))
    return words


class Tokenizer(DummyTokenizer):
    def __init__(
        self,
        cls,
        nlp: Optional[Language] = None,
        juman_kwargs: Optional[Dict[str, str]] = None,
    ):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        if juman_kwargs:
            self.tokenizer = Juman(**juman_kwargs)
        else:
            self.tokenizer = Juman()

    def __call__(self, text):
        dtokens = detailed_tokens(self.tokenizer, text)
        words = [x.surface for x in dtokens]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.lemma_ = dtoken.lemma
            token.tag_ = dtoken.pos
        return doc


class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda _text: "ja_juman"
    stop_words = STOP_WORDS
    tag_map = TAG_MAP
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(cls, nlp=None, juman_kwargs: Optional[Dict[str, Any]] = None):
        return Tokenizer(cls, nlp, juman_kwargs=juman_kwargs)


class Japanese(Language):
    lang = "ja_juman"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)


__all__ = ["Japanese"]
