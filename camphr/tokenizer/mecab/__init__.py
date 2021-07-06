"""The package mecab defines Japanese spacy.Language with Mecab tokenizer."""
from camphr.doc import Doc, DocProto, TokenProto
from typing import List, NamedTuple, TYPE_CHECKING
from camphr.serde import SerDe
from typing_extensions import Literal, Protocol

if TYPE_CHECKING:
    from MeCab import Tagger


class ShortUnitWord(NamedTuple):
    surface: str
    lemma: str
    pos: str
    space: str
    fstring: str


def get_dictionary_type(
    tagger: "Tagger",
) -> Literal["ipadic", "unidic", "neologd", "juman"]:
    filename = tagger.dictionary_info().filename  # type: ignore
    for k in ["ipadic", "unidic", "neologd", "juman"]:
        if k in filename:
            return k  # type: ignore
    raise ValueError(f"Unsupported dictionary type: {filename}")


def get_mecab_tagger():
    """Create `MeCab.Tagger` instance"""
    import MeCab  # type: ignore

    tokenizer = MeCab.Tagger()
    tokenizer.parseToNode("")  # type: ignore # see https://github.com/explosion/spaCy/issues/2901
    return tokenizer


class _MecabNode(Protocol):
    next: "_MecabNode"
    surface: str
    posid: int
    feature: str
    length: int
    rlength: int


class Tokenizer(SerDe):
    USERDIC = "user.dic"  # used when saving
    ASSETS = "assets"  # used when saving
    KEY_FSTRING = "mecab_fstring"

    @classmethod
    def get_mecab_fstring(cls, token: TokenProto) -> str:
        return token.user_data[cls.KEY_FSTRING]

    @classmethod
    def set_mecab_fstring(cls, token: TokenProto, fstring: str):
        token.user_data[cls.KEY_FSTRING] = fstring

    def __init__(self):
        self.tokenizer = get_mecab_tagger()
        self.dictionary_type = get_dictionary_type(self.tokenizer)

    def __call__(self, text: str) -> DocProto:
        dtokens = self.detailed_tokens(text)
        words = [x.surface + x.space for x in dtokens]
        doc = Doc.from_words(words)
        for token, dtoken in zip(doc, dtokens):
            token.tag_ = dtoken.pos
            token.lemma_ = dtoken.lemma if dtoken.lemma != "*" else token.text
            self.set_mecab_fstring(token, dtoken.fstring)
        return doc

    def detailed_tokens(self, text: str) -> List[ShortUnitWord]:
        """Tokenize text with Mecab and format the outputs for further processing"""
        node: _MecabNode = self.tokenizer.parseToNode(text)  # type: ignore
        node = node.next
        if self.dictionary_type == "unidic":
            lemma_idx = 10
        elif self.dictionary_type == "juman":
            lemma_idx = 4
        else:
            lemma_idx = 6
        words: List[ShortUnitWord] = []
        while node.posid != 0:
            parts: List[str] = node.feature.split(",")
            pos = ",".join(parts[0:4])
            surface = node.surface
            base = parts[lemma_idx] if len(parts) > lemma_idx else surface
            nextnode = node.next
            # TODO: make non-destructive
            if nextnode.length != nextnode.rlength:
                # next node contains space, so attach it to this node.
                words.append(ShortUnitWord(surface, base, pos, " ", node.feature))
            elif nextnode.surface == "\u3000":
                # next node is full space, so attatch it to this node and skip the nextnode.
                words.append(
                    ShortUnitWord(surface, base, pos, nextnode.surface, node.feature)
                )
                nextnode = nextnode.next
            else:
                words.append(ShortUnitWord(surface, base, pos, "", node.feature))
            node = nextnode
        return words
