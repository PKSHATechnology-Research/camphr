"""The package juman defines Japanese spacy.Language with JUMAN tokenizer."""
import itertools
from collections import namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Type

from spacy.compat import copy_reg
from spacy.language import Language
from spacy.tokens import Doc, Token

from camphr.consts import JUMAN_LINES, KEY_FSTRING
from camphr.lang.stop_words import STOP_WORDS
from camphr.utils import SerializationMixin, get_juman_command

ShortUnitWord = namedtuple(
    "ShortUnitWord", ["surface", "lemma", "pos", "fstring", "space"]
)


def han_to_zen_normalize(text):
    try:
        import mojimoji
    except ImportError:
        raise ValueError("juman or knp Language requires mojimoji.")
    return mojimoji.han_to_zen(text.replace("\t", " ").replace("\r", ""))


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
        cls: Type["Defaults"],
        nlp: Optional[Language] = None,
        juman_kwargs: Optional[Dict[str, str]] = None,
        preprocessor: Optional[Callable[[str], str]] = han_to_zen_normalize,
    ):
        """

        Args:
            juman_kwargs: passed to `pyknp.Juman.__init__`
            preprocessor: applied to text before tokenizing. `mojimoji.han_to_zen` is often used.
        """
        from pyknp import Juman

        juman_kwargs = juman_kwargs or {}
        default_command = get_juman_command()
        assert default_command
        juman_kwargs.setdefault("command", default_command)

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
        juman_lines = self._juman_string(text)
        dtokens = self._detailed_tokens(juman_lines)
        doc = self._dtokens_to_doc(dtokens)
        doc.user_data[JUMAN_LINES] = juman_lines
        return doc

    def _juman_string(self, text: str) -> str:
        try:
            texts = _split_text_for_juman(text)
            lines: str = "".join(
                itertools.chain.from_iterable(
                    self.tokenizer.juman_lines(text) for text in texts
                )
            )
        except BrokenPipeError:
            # Juman is sometimes broken due to its subprocess management.
            self.reset_tokenizer()
            lines = self.tokenizer.juman_lines(text)
        return lines

    def _dtokens_to_doc(self, dtokens: List[ShortUnitWord]) -> Doc:
        words = [x.surface for x in dtokens]
        spaces = [x.space for x in dtokens]
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for token, dtoken in zip(doc, dtokens):
            token.lemma_ = dtoken.lemma
            token.tag_ = dtoken.pos
            token._.set(self.key_fstring, dtoken.fstring)
        doc.is_tagged = True
        return doc

    def _detailed_tokens(self, juman_lines: str) -> List[ShortUnitWord]:
        """Tokenize text with Juman and format the outputs for further processing"""
        from pyknp import MList

        ml = MList(juman_lines).mrph_list()
        words: List[ShortUnitWord] = []
        for m in ml:
            surface = m.midasi
            pos = m.hinsi + "," + m.bunrui
            lemma = m.genkei or surface
            words.append(ShortUnitWord(surface, lemma, pos, m.fstring, False))
        return words


_SEPS = ["。", ".", "．"]


def _split_text_for_juman(text: str) -> Iterator[str]:
    """Juman denies long text (maybe >4096 bytes) so split text"""
    n = 1000
    if len(text) < n:
        yield text
        return
    for sep in _SEPS:
        if sep in text:
            i = text.index(sep)
            head, tail = text[: i + 1], text[i + 1 :]
            if len(head) < n:
                yield from _split_text_for_juman(head)
                yield from _split_text_for_juman(tail)
                return
    # If any separator is not found in text, split roughly
    yield text[:n]
    yield from _split_text_for_juman(text[n:])


# for pickling. see https://spacy.io/usage/adding-languages
class Defaults(Language.Defaults):  # type: ignore
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
    lang = "ja_juman"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


# avoid pickling problem (see https://github.com/explosion/spaCy/issues/3191)
def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)
Language.factories[Japanese.lang] = Japanese

# for lazy loading. see https://spacy.io/usage/adding-languages
__all__ = ["Japanese"]

Tokenizer.install_extensions()
