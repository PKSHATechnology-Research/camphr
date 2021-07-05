"""The package juman defines Japanese spacy.Language with JUMAN tokenizer."""
from dataclasses import dataclass
from camphr.doc import Doc, DocProto
import itertools
from typing import Any, Callable, Dict, Iterator, List, Optional, Type


from camphr.utils import get_juman_command
from camphr.serde import SerializationMixin


@dataclass
class ShortUnitWord:
    surface: str
    lemma: str
    pos: str
    fstring: str
    space: str


_REPLACE_STRINGS = {"\t": "　", "\r": "", "\n": "　"}


def han_to_zen_normalize(text: str):
    try:
        import mojimoji
    except ImportError:
        raise ValueError("juman or knp Language requires mojimoji.")
    text = mojimoji.han_to_zen(text)
    for k, v in _REPLACE_STRINGS.items():
        text = text.replace(k, v)
    return text


class Tokenizer(SerializationMixin):
    """Juman tokenizer

    Note:
        `spacy.Token._.fstring` is set. The Juman's output is stored into it during tokenizing.
    """

    serialization_fields = ["preprocessor", "juman_kwargs"]
    KEY_FSTRING = "juman_fstring"

    @classmethod
    def get_juman_fstring(cls, doc: DocProto[Any]) -> str:
        if cls.KEY_FSTRING not in doc.user_data:
            raise ValueError("`doc` is not parsed by juman.")
        return doc.user_data[cls.KEY_FSTRING]

    def __init__(
        self,
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
            token.tag_ = dtoken.pos
            token.lemma_ = dtoken.lemma
            token._.set(self.KEY_FSTRING, dtoken.fstring)
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
    if len(text) == 0:
        return
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
