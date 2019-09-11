from typing import Optional, List, Dict, Any, Callable
import re
from collections import namedtuple

from bedoner.consts import KEY_FSTRING, KEY_KNP_ENT, KEY_KNP_ENT_IOB
from pyknp import KNP, Morpheme
from bedoner.lang.stop_words import STOP_WORDS
from spacy.attrs import LANG
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.compat import copy_reg
from spacy.util import DummyTokenizer


ShortUnitWord = namedtuple(
    "ShortUnitWord", ["surface", "lemma", "pos", "fstring", "ent", "ent_iob"]
)

LOC2IOB = {"B": "B", "I": "I", "E": "I", "S": "B"}


def detailed_tokens(tokenizer: KNP, text) -> List[Morpheme]:
    """Format juman output for tokenizing"""
    words = []
    try:
        ml = tokenizer.parse(text).mrph_list()
    except:
        raise ValueError(f"Cannot parse '{text}'")

    for m in ml:
        # m: Morpheme = m
        surface = m.midasi
        pos = m.hinsi + "/" + m.bunrui
        lemma = m.genkei or surface

        ent, iob = "", ""
        ents = re.findall(r"\<NE\:(\w+)\:(.*)?\>", m.fstring)
        if ents:
            ent, loc = ents[0]
            iob = LOC2IOB[loc]

        words.append(ShortUnitWord(surface, lemma, pos, m.fstring, ent, iob))
    return words


class Tokenizer(DummyTokenizer):
    """knp tokenizer"""

    def __init__(
        self,
        cls,
        nlp: Optional[Language] = None,
        knp_kwargs: Optional[Dict[str, str]] = None,
        preprocessor: Callable[[str], str] = None,
    ):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        if knp_kwargs:
            self.tokenizer = KNP(**knp_kwargs)
        else:
            self.tokenizer = KNP()

        self.key_fstring = KEY_FSTRING
        self.key_ent = KEY_KNP_ENT
        self.key_ent_iob = KEY_KNP_ENT_IOB
        Token.set_extension(self.key_fstring, default="", force=True)
        Token.set_extension(self.key_ent, default="", force=True)
        Token.set_extension(self.key_ent_iob, default="", force=True)
        self.preprocessor = preprocessor

    def __call__(self, text):
        if self.preprocessor:
            text = self.preprocessor(text)
        dtokens = detailed_tokens(self.tokenizer, text)
        words = [x.surface for x in dtokens]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.lemma_ = dtoken.lemma
            token.tag_ = dtoken.pos
            token._.set(self.key_fstring, dtoken.fstring)
            token._.set(self.key_ent, dtoken.ent)
            token._.set(self.key_ent_iob, dtoken.ent_iob)
        return doc


class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda _text: "knp"
    stop_words = STOP_WORDS
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(cls, nlp=None, knp_kwargs: Optional[Dict[str, Any]] = None):
        return Tokenizer(cls, nlp, knp_kwargs=knp_kwargs)


class Japanese(Language):
    lang = "knp"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)


__all__ = ["Japanese"]
