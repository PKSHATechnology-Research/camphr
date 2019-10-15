"""The package knp defines Japanese spacy.Language with knp tokenizer."""
import re
from collections import namedtuple
from itertools import tee, zip_longest
from typing import Any, Callable, Dict, List, Optional

from spacy.attrs import LANG
from spacy.compat import copy_reg
from spacy.language import Language
from spacy.tokens import Doc, Token

from bedoner.consts import KEY_FSTRING, KEY_KNP_ENT, KEY_KNP_ENT_IOB
from bedoner.lang.stop_words import STOP_WORDS
from bedoner.utils import SerializationMixin

ShortUnitWord = namedtuple(
    "ShortUnitWord", ["surface", "lemma", "pos", "fstring", "ent", "ent_iob", "space"]
)

LOC2IOB = {"B": "B", "I": "I", "E": "I", "S": "B"}


class Tokenizer(SerializationMixin):
    """KNP tokenizer

    Note:
        `spacy.Token._.fstring` is set in init, and store the KNP's output into it during tokenizing.
        `spacy.Token._.knp_ent` is set in init, and store the entity label KNP parsed into it during tokenizing.
        `spacy.Token._.knp_ent_iob` is set in init, and store the entity iob KNP parsed into it during tokenizing.
    """

    key_fstring = KEY_FSTRING
    key_ent = KEY_KNP_ENT
    key_ent_iob = KEY_KNP_ENT_IOB

    @classmethod
    def install_extensions(cls):
        Token.set_extension(cls.key_fstring, default=None, force=True)
        Token.set_extension(cls.key_ent, default=None, force=True)
        Token.set_extension(cls.key_ent_iob, default=None, force=True)

    def __init__(
        self,
        cls: Language,
        nlp: Optional[Language] = None,
        knp_kwargs: Optional[Dict[str, str]] = None,
        preprocessor: Callable[[str], str] = None,
    ):
        """

        Args:
            knp_kwargs: passed to `pyknp.KNP.__init__`
            preprocessor: applied to text before tokenizing. `mojimoji.han_to_zen` is often used.
        """
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        from pyknp import KNP

        self.tokenizer = KNP(**knp_kwargs) if knp_kwargs else KNP()
        self.knp_kwargs = knp_kwargs

        self.preprocessor = preprocessor

    def __call__(self, text: str) -> Doc:
        if self.preprocessor:
            text = self.preprocessor(text)
        text = text.replace(" ", "\u3000")
        dtokens = self.detailed_tokens(text)
        words = [x.surface for x in dtokens]
        spaces = [x.space for x in dtokens]
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.lemma_ = dtoken.lemma
            token.tag_ = dtoken.pos
            token._.set(self.key_fstring, dtoken.fstring)
            token._.set(self.key_ent, dtoken.ent)
            token._.set(self.key_ent_iob, dtoken.ent_iob)
        doc.is_tagged = True
        return doc

    def detailed_tokens(self, text: str) -> List[ShortUnitWord]:
        """Tokenize text with KNP and format the outputs for further processing"""
        from pyknp import Morpheme

        words: List[Morpheme] = []
        ml: List[Morpheme] = self.tokenizer.parse(text).mrph_list()
        morphs, next_morphs = tee(ml)
        next(next_morphs)
        for m, nextm in zip_longest(morphs, next_morphs):
            if is_space_morph(m):
                continue
            surface = m.midasi
            pos = m.hinsi + "," + m.bunrui
            lemma = m.genkei or surface

            ent, iob = "", ""
            ents = re.findall(r"\<NE\:(\w+)\:(.*)?\>", m.fstring)
            if ents:
                ent, loc = ents[0]
                iob = LOC2IOB[loc]
            if nextm and is_space_morph(nextm):
                words.append(
                    ShortUnitWord(surface, lemma, pos, m.fstring, ent, iob, True)
                )
            else:
                words.append(
                    ShortUnitWord(surface, lemma, pos, m.fstring, ent, iob, False)
                )
        return words


def is_space_morph(m) -> bool:
    return m.bunrui == "空白"


# for pickling. see https://spacy.io/usage/adding-languages
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


# avoid pickling problem (see https://github.com/explosion/spaCy/issues/3191)
def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)


# for lazy loading. see https://spacy.io/usage/adding-languages
__all__ = ["Japanese"]

Tokenizer.install_extensions()
