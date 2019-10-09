"""Module wordpiecer defines wordpiecer for pytorch transformers."""
from typing import Iterable, List

import transformers as trf
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy_transformers.pipeline.wordpiecer import TransformersWordPiecer, get_tokenizer

BERT_PRETRAINED_VOCAB_ARCHIVE_MAP = {
    "bert-ja-juman": "s3://bedoner/pytt_models/bert/bert-ja-juman-vocab.txt"  # bedore-ranndd aws account
}
PRETRAINED_INIT_CONFIGURATION = {
    "bert-ja-juman": {
        "do_lower_case": False,
        "tokenize_chinese_chars": False,
    }  # do_lower_case=False: 濁点落ちを防ぐ，tokenize_chinese_chars=False: スペース以外のspiltを防ぐ
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"bert-ja-juman": 512}


class WordPiecer(TransformersWordPiecer):
    name = "trf_wordpiecer"

    @classmethod
    def from_pretrained(cls, vocab: Vocab, pytt_name: str, **cfg):
        """Instantiate wordpiecer from pretrained model.

        Examples:
            >>> nlp = PyttWordPiecer.from_pretrained("bert-ja-juman")
        """
        pytt_tokenizer_cls: trf.PreTrainedTokenizer = get_tokenizer(pytt_name)

        # tell `pytt_tokenizer_cls` where to find the model
        pytt_tokenizer_cls.pretrained_vocab_files_map["vocab_file"].update(
            BERT_PRETRAINED_VOCAB_ARCHIVE_MAP
        )
        # tell `pytt_tokenizer_cls` how to configure the model
        pytt_tokenizer_cls.pretrained_init_configuration.update(
            PRETRAINED_INIT_CONFIGURATION
        )
        pytt_tokenizer_cls.max_model_input_sizes.update(
            PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        )
        model = pytt_tokenizer_cls.from_pretrained(pytt_name)
        return cls(vocab, model=model, pytt_name=pytt_name, **cfg)

    def update(self, docs: Iterable[Doc], *args, **kwargs) -> List[Doc]:
        """Simply forward docs. This method is called when `spacy.Language.update`."""
        return [self(doc) for doc in docs]
