from typing import Optional, List, Dict, Any
from collections import namedtuple

from bedoner.consts import KEY_FSTRING
from pyknp import Juman, Morpheme
from bedoner.lang.stop_words import STOP_WORDS
from spacy.attrs import LANG
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.compat import copy_reg
from spacy.util import DummyTokenizer


ShortUnitWord = namedtuple("ShortUnitWord", ["surface", "lemma", "pos", "fstring"])


def detailed_tokens(tokenizer: Juman, text) -> List[Morpheme]:
    """Format juman output for tokenizing"""
    words = []
    ml = tokenizer.analysis(text).mrph_list()
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
        juman_kwargs: Optional[Dict[str, str]] = None,
    ):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        if juman_kwargs:
            self.tokenizer = Juman(**juman_kwargs)
        else:
            self.tokenizer = Juman()

        self.key_fstring = KEY_FSTRING
        Token.set_extension(self.key_fstring, default=False, force=True)

    def __call__(self, text):
        dtokens = detailed_tokens(self.tokenizer, text)
        words = [x.surface for x in dtokens]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.lemma_ = dtoken.lemma
            token.tag_ = dtoken.pos
            token._.set(self.key_fstring, dtoken.fstring)
        return doc


class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda _text: "ja_juman"
    stop_words = STOP_WORDS
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
