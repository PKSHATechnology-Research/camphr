"""Module wordpiecer defines wordpiecer for pytorch transformers."""
from typing import Iterable, List, Type

import transformers as trf
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy_transformers.pipeline.wordpiecer import TransformersWordPiecer, get_tokenizer
from spacy_transformers.util import ATTRS

from bedoner.lang.sentencepiece import EXTS

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    "bert-ja-juman": "s3://bedoner/trf_models/bert/bert-ja-juman-vocab.txt",  # bedore-ranndd aws account
    "xlnet-ja": "s3://bedoner/trf_models/xlnet/spiece.model",
}
PRETRAINED_INIT_CONFIGURATION = {
    "bert-ja-juman": {
        "do_lower_case": False,
        "tokenize_chinese_chars": False,
    },  # do_lower_case=False: 濁点落ちを防ぐ，tokenize_chinese_chars=False: スペース以外のspiltを防ぐ
    "xlnet-ja": {"keep_accesnts": True},
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"bert-ja-juman": 512}


class WordPiecer(TransformersWordPiecer):
    name = "trf_wordpiecer"

    @classmethod
    def from_pretrained(cls, vocab: Vocab, trf_name: str, **cfg):
        """Instantiate wordpiecer from pretrained model.

        Examples:
            >>> nlp = WordPiecer.from_pretrained("bert-ja-juman")
        """
        trf_tokenizer_cls: Type[trf.PreTrainedTokenizer] = get_tokenizer(trf_name)

        # tell `trf_tokenizer_cls` where to find the model
        trf_tokenizer_cls.pretrained_vocab_files_map["vocab_file"].update(
            PRETRAINED_VOCAB_ARCHIVE_MAP
        )
        # tell `trf_tokenizer_cls` how to configure the model
        trf_tokenizer_cls.pretrained_init_configuration.update(
            PRETRAINED_INIT_CONFIGURATION
        )
        trf_tokenizer_cls.max_model_input_sizes.update(
            PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        )
        model = trf_tokenizer_cls.from_pretrained(trf_name)
        return cls(vocab, model=model, trf_name=trf_name, **cfg)

    def update(self, docs: Iterable[Doc], *args, **kwargs) -> List[Doc]:
        """Simply forward docs. This method is called when `spacy.Language.update`."""
        return [self(doc) for doc in docs]


class TrfSentencePiecer(TransformersWordPiecer):
    name = "trf_sentencepiecer"

    def __init__(self, vocab, model=True, **cfg):
        self.vocab = vocab
        self.model = model
        self.cfg = cfg

    @classmethod
    def from_pretrained(cls, vocab: Vocab, trf_name: str, **cfg):
        """Instantiate wordpiecer from pretrained model.

        Examples:
            >>> nlp = WordPiecer.from_pretrained("bert-ja-juman")
        """
        trf_tokenizer_cls: Type[trf.PreTrainedTokenizer] = get_tokenizer(trf_name)

        # tell `trf_tokenizer_cls` where to find the model
        trf_tokenizer_cls.pretrained_vocab_files_map["vocab_file"].update(
            PRETRAINED_VOCAB_ARCHIVE_MAP
        )
        # tell `trf_tokenizer_cls` how to configure the model
        trf_tokenizer_cls.pretrained_init_configuration.update(
            PRETRAINED_INIT_CONFIGURATION
        )
        trf_tokenizer_cls.max_model_input_sizes.update(
            PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        )
        model = trf_tokenizer_cls.from_pretrained(trf_name)
        return cls(vocab, model=model, trf_name=trf_name, **cfg)

    def predict(self, docs: Iterable[Doc]) -> List[List[str]]:
        # TODO: align, segment
        pieces = []
        for doc in docs:
            pieces.append(self.model.add_special_tokens([doc._.get(EXTS.pieces_)]))
        return pieces

    def set_annotations(
        self, docs: Iterable[Doc], predictions: List[List[str]]
    ) -> Iterable[Doc]:
        for doc, piece in zip(docs, predictions):
            doc._.set(ATTRS.alignment, doc._.get(EXTS.alignment))
            doc._.set(ATTRS.word_pieces, self.model.convert_tokens_to_ids(piece))
            doc._.set(ATTRS.word_pieces_, piece)
        return docs

    def update(self, docs: Iterable[Doc], *args, **kwargs) -> Iterable[Doc]:
        """Simply forward docs. This method is called when `spacy.Language.update`."""
        return docs
