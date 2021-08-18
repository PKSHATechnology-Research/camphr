"""The package juman defines Japanese spacy.Language with JUMAN tokenizer."""
from dataclasses import dataclass
from camphr.doc import Doc, UserDataProto
import itertools
import distutils.spawn
from typing import Any, Callable, Dict, Iterator, List, Optional
from typing_extensions import Literal


from camphr.serde import SerializationMixin


def get_juman_command() -> Optional[Literal["juman", "jumanpp"]]:
    for cmd in ["jumanpp", "juman"]:
        if distutils.spawn.find_executable(cmd):
            return cmd  # type: ignore
    return None


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
    def get_juman_fstring(cls, e: UserDataProto) -> str:
        if cls.KEY_FSTRING not in e.user_data:
            raise ValueError(f"{cls.KEY_FSTRING} is not set in {e}")
        return e.user_data[cls.KEY_FSTRING]

    @classmethod
    def set_juman_fstring(cls, e: UserDataProto, fstring: str):
        e.user_data[cls.KEY_FSTRING] = fstring

    def __init__(
        self,
        juman_kwargs: Optional[Dict[str, Any]] = None,
        preprocessor: Optional[Callable[[str], str]] = han_to_zen_normalize,
    ):
        """

        Args:
            juman_kwargs: passed to `pyknp.Juman.__init__`
            preprocessor: applied to text before tokenizing. `mojimoji.han_to_zen` is often used.
        """

        juman_kwargs = juman_kwargs or {}
        default_command = get_juman_command()
        assert default_command
        juman_kwargs.setdefault("command", default_command)

        self.juman_kwargs = juman_kwargs
        self.preprocessor = preprocessor
        self.set_tokenizer()

    def set_tokenizer(self):
        from pyknp import Juman

        self.tokenizer = Juman(**self.juman_kwargs) if self.juman_kwargs else Juman()

    def __call__(self, text: str) -> Doc:
        """Make doc from text. Juman's `fstring` is stored in `Token._.fstring`"""
        if self.preprocessor:
            text = self.preprocessor(text)
        juman_lines = self._juman_parse(text)
        dtokens = self._detailed_tokens(juman_lines)
        doc = self._dtokens_to_doc(dtokens)
        self.set_juman_fstring(doc, juman_lines)
        return doc

    def _juman_parse(self, text: str) -> str:
        texts = _split_text_for_juman(text)
        while True:
            try:
                lines: str = "".join(
                    itertools.chain.from_iterable(
                        self.tokenizer.juman_lines(text) for text in texts  # type: ignore
                    )
                )
                break
            except BrokenPipeError:
                # Juman is sometimes broken due to its subprocess management.
                self.set_tokenizer()
        return lines

    def _dtokens_to_doc(self, dtokens: List[ShortUnitWord]) -> Doc:
        words = [x.surface + x.space for x in dtokens]
        doc = Doc.from_words(words)
        for token, dtoken in zip(doc, dtokens):
            token.tag_ = dtoken.pos
            token.lemma_ = dtoken.lemma
            self.set_juman_fstring(token, dtoken.fstring)
        return doc

    def _detailed_tokens(self, juman_lines: str) -> List[ShortUnitWord]:
        """Tokenize text with Juman and format the outputs for further processing"""
        from pyknp import MList, Morpheme  # type: ignore

        ml: List[Morpheme] = MList(juman_lines).mrph_list()
        words: List[ShortUnitWord] = []
        for m in ml:
            surface: str = m.midasi  # type: ignore
            pos: str = m.hinsi + "," + m.bunrui  # type: ignore
            lemma: str = m.genkei or surface  # type: ignore
            words.append(ShortUnitWord(surface, lemma, pos, m.fstring, ""))  # type: ignore
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
