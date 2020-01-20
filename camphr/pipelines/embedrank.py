"""Pipeline of Embed Rank (https://arxiv.org/pdf/1801.04470.pdf).

`EmbedRank` assigns `Doc` extension `embedrank_keyphrases`
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import spacy
import spacy.language
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from camphr.utils import SerializationMixin

spacy.language.ENABLE_PIPELINE_ANALYSIS = True


EMBEDRANK_KEYPHRASES = "embedrank_keyphrases"


class ExtractKeywordsRuler:
    def __init__(self, vocab: Vocab, patterns: Dict[str, List]):
        self.matcher = Matcher(vocab)
        for k, v in patterns.items():
            self.matcher.add(k, None, v)

    def __call__(self, doc: Doc) -> List[Span]:
        return [doc[i:j] for _, i, j in self.matcher(doc)]


@spacy.component(
    "embedrank",
    assigns=[f"doc._.{EMBEDRANK_KEYPHRASES}"],
    requires=["doc.vector", "token.vector"],
)
class EmbedRank(SerializationMixin):
    DefaultPatterns = {
        "keyword": [
            {"TAG": {"REGEX": "(名詞|形容詞|助詞,連体化|接頭詞,名詞接続).*"}, "OP": "*"},
            {"TAG": {"REGEX": "名詞,.*"}, "OP": "+"},
        ]
    }

    serialization_fields = ["max_keyphrases", "extract_keyphrases", "lambda_"]

    @staticmethod
    def install_extensions():
        Doc.set_extension(EMBEDRANK_KEYPHRASES, default=None)

    def __init__(
        self,
        vocab: Optional[Vocab] = None,
        max_keyphrases: int = -1,
        extract_keyphrases_fn: Callable[[Doc], List[Span]] = None,
        lambda_: float = 0.5,
    ):
        """

        Args:
            max_keyphrases: max number of keyphrases set in doc._.embedrank_keyphrases. If -1, all keyphrases are set.
            extract_keyphrases_fn: Function that returns the candidates of keyword.
            lambda_: Hyperparameter of Maximal Marginal Relevance (MMR). See the paper (https://arxiv.org/pdf/1801.04470.pdf) for details.
        """
        self.vocab = vocab
        self.extract_keyphrases = extract_keyphrases_fn
        self.max_keyphrases = max_keyphrases
        self.lambda_ = lambda_

    @classmethod
    def from_nlp(cls, nlp: spacy.language.Language) -> "EmbedRank":
        return cls(nlp.vocab)

    def require_model(self):
        if self.extract_keyphrases is None:
            assert self.vocab is not None
            self.extract_keyphrases = ExtractKeywordsRuler(
                self.vocab, self.DefaultPatterns
            )

    def __call__(self, doc: Doc) -> Doc:
        """Extract keyphrases from doc.vector and span.vector, and set them into Doc._.embed_keyphrases sorted by score."""
        self.require_model()
        spans = self.extract_keyphrases(doc)
        if not spans:
            return doc
        spans_vectors = np.array([np.array(span.vector) for span in spans])
        spans_sims = cosine_similarity(spans_vectors, spans_vectors)
        scores0 = self.lambda_ * cosine_similarity(
            spans_vectors, doc.vector.reshape(1, -1)
        )
        scores0 = np.squeeze(scores0)

        candidates = list(range(len(spans)))
        i = np.argmax(scores0)
        candidates.remove(i)
        populated: List[int] = [i]

        while candidates and (
            self.max_keyphrases <= 0 or len(populated) < self.max_keyphrases
        ):
            scores = scores0[candidates] - (1 - self.lambda_) * np.max(
                spans_sims[candidates][:, populated], axis=1
            )
            i = np.argmax(scores)
            populated.append(candidates[i])
            candidates.pop(i)
        doc._.set(EMBEDRANK_KEYPHRASES, [spans[i] for i in populated])
        return doc


EmbedRank.install_extensions()
