"""The package juman defines Japanese spacy.Language with JUMAN tokenizer."""
from collections import namedtuple
from itertools import tee, zip_longest
from typing import Any, Callable, Dict, List, Optional

from spacy.compat import copy_reg
from spacy.language import Language
from spacy.tokens import Doc, Token

from bedoner.consts import KEY_FSTRING
from bedoner.lang.stop_words import STOP_WORDS
from bedoner.utils import SerializationMixin
from bedoner.lang.torch_mixin import TorchLanguageMixin

ShortUnitWord = namedtuple(
    "ShortUnitWord", ["surface", "lemma", "pos", "fstring", "space"]
)


class Tokenizer(SerializationMixin):
    """Juman tokenizer

    Note:
        `spacy.Token._.fstring` is set. The Juman's output is stored into it during tokenizing.
    """

    serialization_fields = ["preprocessor", "juman_kwargs"]
    key_fstring = KEY_FSTRING

    @classmethod
    def install_extensions(cls):
        """See https://github.com/explosion/spacy-pytorch-transformers#extension-attributes."""
        Token.set_extension(cls.key_fstring, default=None, force=True)

    def __init__(
        self,
        cls: Language,
        nlp: Optional[Language] = None,
        juman_kwargs: Optional[Dict[str, str]] = None,
        preprocessor: Optional[Callable[[str], str]] = None,
    ):
        """

        Args:
            juman_kwargs: passed to `pyknp.Juman.__init__`
            preprocessor: applied to text before tokenizing. `mojimoji.han_to_zen` is often used.
        """
        from pyknp import Juman

        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        self.tokenizer = Juman(**juman_kwargs) if juman_kwargs else Juman()
        self.juman_kwargs = juman_kwargs
        self.preprocessor = preprocessor

    def reset_tokenizer(self):
        from pyknp import Juman

        self.tokenizer = Juman(**self.juman_kwargs) if self.juman_kwargs else Juman()

    def __call__(self, text: str) -> Doc:
        """Make doc from text. Juman's `fstring` is stored in `Token._.fstring`"""
        if self.preprocessor:
            text = self.preprocessor(text)
        dtokens = self.detailed_tokens(text)
        words = [x.surface for x in dtokens]
        spaces = [x.space for x in dtokens]
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.lemma_ = dtoken.lemma
            token.tag_ = dtoken.pos
            token._.set(self.key_fstring, dtoken.fstring)
        doc.is_tagged = True
        return doc

    def detailed_tokens(self, text: str) -> List[ShortUnitWord]:
        """Tokenize text with Juman and format the outputs for further processing"""
        from pyknp import Morpheme

        words: List[ShortUnitWord] = []
        try:
            ml: List[Morpheme] = self.tokenizer.analysis(text).mrph_list()
        except BrokenPipeError:
            # Juman is sometimes broken due to its subprocess management.
            self.reset_tokenizer()
            ml: List[Morpheme] = self.tokenizer.analysis(text).mrph_list()
        morphs, next_morphs = tee(ml)
        next(next_morphs)
        for m, nextm in zip_longest(morphs, next_morphs):
            if is_space_morph(m):
                continue
            surface = m.midasi
            pos = m.hinsi + "," + m.bunrui
            lemma = m.genkei or surface
            if nextm and is_space_morph(nextm):
                words.append(ShortUnitWord(surface, lemma, pos, m.fstring, True))
            else:
                words.append(ShortUnitWord(surface, lemma, pos, m.fstring, False))
        return words


def is_space_morph(m) -> bool:
    return m.bunrui == "空白"


# for pickling. see https://spacy.io/usage/adding-languages
class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    stop_words = STOP_WORDS
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(
        cls,
        nlp=None,
        juman_kwargs: Optional[Dict[str, Any]] = None,
        preprocessor: Optional[Callable[[str], str]] = None,
    ):
        return Tokenizer(cls, nlp, juman_kwargs=juman_kwargs, preprocessor=preprocessor)


class Japanese(Language):
    lang = "juman"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


# avoid pickling problem (see https://github.com/explosion/spaCy/issues/3191)
def pickle_japanese(instance):
    return Japanese, tuple()


class TorchJapanese(TorchLanguageMixin, Japanese):
    lang = "torch_juman"


copy_reg.pickle(Japanese, pickle_japanese)
Language.factories[Japanese.lang] = Japanese
Language.factories[TorchJapanese.lang] = TorchJapanese

# for lazy loading. see https://spacy.io/usage/adding-languages
__all__ = ["Japanese"]

Tokenizer.install_extensions()
