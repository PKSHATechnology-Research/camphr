"""Defines pattern search pipeline based on ahocorasik."""
import itertools
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import ahocorasick
from spacy.tokens.token import Token
import textspan
from spacy.tokens import Doc, Span


# Sometimes matched text is different from original text
# since `PatternSearcher` can match the `lemma`.
# This extension holds the matched text.
PATTERN_MATCH_AS = "pattern_match_as"
Span.set_extension(PATTERN_MATCH_AS, default=None, force=True)


class PatternSearcher:
    def __init__(
        self,
        model: ahocorasick.Automaton,
        label: str = "matched",
        lower: bool = False,
        normalizer: Optional[Callable[[Token], str]] = None,
        lemma: bool = False,
        extend_span_to_token_boundary: bool = False,
        ignore_space: bool = True,
    ):
        self.model = model
        self.lower = lower
        self.normalizer = normalizer
        self.lemma = lemma
        self.label = label
        self.extend_span_to_token_boundary = extend_span_to_token_boundary
        self.ignore_space = ignore_space

    @staticmethod
    def get_model_from_words(words: Iterable[str]) -> ahocorasick.Automaton:
        model = ahocorasick.Automaton()
        for word in words:
            model.add_word(word, word)
        model.make_automaton()
        return model

    def get_char_spans(self, text: str) -> Iterator[Tuple[int, int, str]]:
        for j, word in self.model.iter(text):
            i = j - len(word) + 1
            yield i, j + 1, word

    def _get_spans_from_matches(
        self, matches: Iterable[Tuple[int, int, str]], tokens: List[Tuple[int, int]]
    ) -> Iterator[Tuple[int, int]]:
        idx = sorted((i, j) for i, j, _ in matches)
        spans = textspan.lift_spans_index(idx, tokens)
        for (l, l_ok), (r, r_ok) in spans:
            if self.extend_span_to_token_boundary or (l_ok and r_ok):
                yield (l, r)

    def _search_by_normalizer(
        self, doc: Doc, normalizer: Callable[[Token], str], ignore_space: bool
    ) -> Iterable[Tuple[int, int]]:
        text = ""
        token_spans: List[Tuple[int, int]] = []
        left = 0
        for token in doc:
            token_text = normalizer(token)
            text += token_text
            right = left + len(token_text)
            if not ignore_space:
                text += token.whitespace_
                right += len(token.whitespace_)
            token_spans.append((left, right))
            left = right
        matches = self.get_char_spans(text)
        return self._get_spans_from_matches(matches, token_spans)

    def __call__(self, doc: Doc) -> Doc:
        normalizers: List[Callable[[Token], str]] = [lambda x: x.text]
        if self.lower:
            normalizers.append(lambda x: x.text.lower())
        if self.lemma:
            normalizers.append(lambda x: x.lemma_)
        if self.normalizer is not None:
            normalizers.append(self.normalizer)

        spans: Iterable[Tuple[int, int]] = []
        for normalizer in normalizers:
            spans = itertools.chain(
                spans, self._search_by_normalizer(doc, normalizer, ignore_space=False)
            )
            if self.ignore_space:
                spans = itertools.chain(
                    spans,
                    self._search_by_normalizer(doc, normalizer, ignore_space=True),
                )

        ents = list(doc.ents)
        for i, j in spans:
            ent = Span(doc, i, j, label=self.label)
            ents.append(ent)
        selected = textspan.remove_span_overlaps_idx([(s.start, s.end) for s in ents])
        doc.ents = tuple(ents[i] for i in selected)
        return doc
