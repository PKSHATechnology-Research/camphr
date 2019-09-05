from typing import Optional, List
from collections import namedtuple

import MeCab
from bedoner.lang.stop_words import STOP_WORDS
from .tag_map import TAG_MAP
from spacy.attrs import LANG
from spacy.language import Language
from spacy.tokens import Doc
from spacy.compat import copy_reg
from spacy.util import DummyTokenizer


ShortUnitWord = namedtuple("ShortUnitWord", ["surface", "lemma", "pos"])


def resolve_pos(token):
    """TODO"""
    return token.pos


def detailed_tokens(tokenizer: MeCab.Tagger, text: str) -> List[ShortUnitWord]:
    node = tokenizer.parseToNode(text)
    node = node.next  # first node is beginning of sentence and empty, skip it
    words = []
    while node.posid != 0:
        surface = node.surface
        base = surface  # a default value. Updated if available later.
        parts = node.feature.split(",")
        pos = ",".join(parts[0:4])
        if len(parts) > 6:
            # this information is only available for words in the tokenizer
            # dictionary
            base = parts[6]
        words.append(ShortUnitWord(surface, base, pos))
        node = node.next
    return words


class Tokenizer(DummyTokenizer):
    def __init__(self, cls, nlp: Optional[Language] = None, opt=""):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        self.tokenizer = MeCab.Tagger(opt)
        self.tokenizer.parseToNode("")  # see #2901

    def __call__(self, text):
        dtokens = detailed_tokens(self.tokenizer, text)
        words = [x.surface for x in dtokens]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        mecab_tags = []
        for token, dtoken in zip(doc, dtokens):
            mecab_tags.append(dtoken.pos)
            token.tag_ = resolve_pos(dtoken)
            token.lemma_ = dtoken.lemma
        doc.user_data["mecab_tags"] = mecab_tags
        return doc


class Defaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda _text: "ja"
    stop_words = STOP_WORDS
    tag_map = TAG_MAP
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

    @classmethod
    def create_tokenizer(cls, nlp=None, opt: str = ""):
        return Tokenizer(cls, nlp, opt=opt)


class Japanese(Language):
    lang = "ja"
    Defaults = Defaults

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)


def pickle_japanese(instance):
    return Japanese, tuple()


copy_reg.pickle(Japanese, pickle_japanese)


__all__ = ["Japanese"]
