"""Module wordpiecer defines wordpiecer for pytorch transformers."""
from typing import Iterable, List, Tuple

import spacy
from spacy.tokens import Doc
from spacy_transformers.pipeline.wordpiecer import TransformersWordPiecer
from spacy_transformers.util import ATTRS, get_sents

from camphr.lang.sentencepiece import EXTS

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    "bert-ja-juman": "s3://camphr/trf_models/bert/bert-ja-juman-vocab.txt",  # bedore-ranndd aws account
    "xlnet-ja": "s3://camphr/trf_models/xlnet/spiece.model",
}
PRETRAINED_INIT_CONFIGURATION = {
    "bert-ja-juman": {
        "do_lower_case": False,
        "tokenize_chinese_chars": False,
    },  # do_lower_case=False: 濁点落ちを防ぐ，tokenize_chinese_chars=False: スペース以外のspiltを防ぐ
    "xlnet-ja": {"keep_accesnts": True},
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"bert-ja-juman": 512}


class PIPES:
    transformers_wordpiecer = "transformers_wordpiecer"
    transformers_sentencepiecer = "transformers_sentencepiecer"


@spacy.component(PIPES.transformers_wordpiecer)
class WordPiecer(TransformersWordPiecer):
    def update(self, docs: Iterable[Doc], *args, **kwargs) -> List[Doc]:
        """Simply forward docs. This method is called when `spacy.Language.update`."""
        return [self(doc) for doc in docs]


@spacy.component(PIPES.transformers_sentencepiecer)
class TrfSentencePiecer(TransformersWordPiecer):
    def __init__(self, vocab, model=True, **cfg):
        self.vocab = vocab
        self.model = model
        self.cfg = cfg

    def predict(self, docs: Iterable[Doc]) -> List[Tuple[List[str], List[int]]]:
        """

        Note:
            Override super().predict because docs are already sentencepieced.
        """
        output = []
        for doc in docs:
            doc_words = []
            doc_align = []
            offset = 0
            for sent in get_sents(doc):
                sent_words = []
                sent_align = []
                for segment in sent._.get(ATTRS.segments):
                    seg_words = list(segment._.get(EXTS.pieces_))
                    diff = offset - segment._.get(EXTS.alignment)[0][0]
                    seg_align = [
                        [diff + i for i in l] for l in segment._.get(EXTS.alignment)
                    ]
                    assert len(segment) == len(seg_align)
                    sent_words.append(seg_words)
                    sent_align.append(seg_align)
                sw_flat = self.model.add_special_tokens(sent_words)
                sa_flat = self.model.fix_alignment(sent_align)
                doc_words.extend(sw_flat)
                doc_align.extend(sa_flat)
                offset += len(sw_flat)
            output.append((doc_words, doc_align))
        return output

    def set_annotations(
        self, docs: Iterable[Doc], predictions: List[Tuple[List[str], List[int]]]
    ) -> Iterable[Doc]:
        """

        Note:
            Override super().set_annotations because docs are already sentencepieced.
        """
        for doc, (pieces, align) in zip(docs, predictions):
            doc._.set(ATTRS.alignment, align)
            doc._.set(ATTRS.word_pieces_, pieces)
            doc._.set(ATTRS.word_pieces, self.model.convert_tokens_to_ids(pieces))

            # assersion test
            nr_word = len(doc._.get(ATTRS.word_pieces))
            words_per_sent = sum(
                len(sent._.get(ATTRS.word_pieces)) for sent in get_sents(doc)
            )
            if nr_word != words_per_sent:
                print([repr(w.text) for w in doc])
                for sent in get_sents(doc):
                    print(sent._.get(ATTRS.word_pieces_))
                    for w in sent:
                        print(w.text, w._.get(ATTRS.alignment))
                print(doc._.get(ATTRS.word_pieces_))
                raise ValueError(
                    f"Error calculating word pieces for sentences. Total number "
                    f"of wordpieces in the doc was {nr_word}, but adding up the "
                    f"wordpieces for its sentences we get {words_per_sent}. "
                    f"The doc is: {doc.text}"
                )
        return docs

    def update(self, docs: Iterable[Doc], *args, **kwargs) -> Iterable[Doc]:
        """Simply forward docs. This method is called when `spacy.Language.update`."""
        outputs = self.predict(docs)
        self.set_annotations(docs, outputs)
        return docs
