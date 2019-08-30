import re
from typing import Optional, List, Dict, Any
from collections import namedtuple

from pyknp import KNP, Morpheme
from bedoner.lang.stop_words import STOP_WORDS
from spacy.attrs import LANG
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.compat import copy_reg
from spacy.util import DummyTokenizer


ShortUnitWord = namedtuple("ShortUnitWord", ["surface", "lemma", "pos", "fstring"])


def detailed_tokens(tokenizer: KNP, text) -> List[Morpheme]:
    """Format juman output for tokenizing"""
    words = []
    ml = tokenizer.parse(text).mrph_list()
    for m in ml:
        # m: Morpheme = m
        surface = m.midasi
        pos = m.hinsi + "/" + m.bunrui
        lemma = m.genkei or surface
        words.append(ShortUnitWord(surface, lemma, pos, m.fstring))
    return words


class Tokenizer(DummyTokenizer):
    """knp tokenizer"""

    def __init__(
        self,
        cls,
        nlp: Optional[Language] = None,
        knp_kwargs: Optional[Dict[str, str]] = None,
        key_fstring: str = "fstring",
    ):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        if knp_kwargs:
            self.tokenizer = KNP(**knp_kwargs)
        else:
            self.tokenizer = KNP()
        self.key_fstring = key_fstring
        Token.set_extension(key_fstring, default="")

    def __call__(self, text):
        dtokens = detailed_tokens(self.tokenizer, text)
        words = [x.surface for x in dtokens]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.lemma_ = dtoken.lemma
            token.tag_ = dtoken.pos
            token.set(self.key_fstring, dtoken.fstring)
        return doc


class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda _text: "ja_juman"
    stop_words = STOP_WORDS
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(cls, nlp=None, knp_kwargs: Optional[Dict[str, Any]] = None):
        return Tokenizer(cls, nlp, knp_kwargs=knp_kwargs)


class Japanese(Language):
    lang = "ja_knp"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)


__all__ = ["Japanese"]
