from pathlib import Path
from spacy_pytorch_transformers.pipeline.wordpiecer import PyTT_WordPiecer
from spacy_pytorch_transformers._tokenizers import SerializableBertTokenizer
from spacy.language import Language


class BertWordPiecer(PyTT_WordPiecer):
    name = "bert_wordpiecer"

    @classmethod
    def Model(cls, vocab_file: str, **kwargs) -> SerializableBertTokenizer:
        model = SerializableBertTokenizer(
            vocab_file=vocab_file, do_lower_case=False, tokenize_chinese_chars=False
        )  # do_lower_case=False: 濁点落ちを防ぐ，tokenize_chinese_chars=False: スペース以外のspiltを防ぐ
        return model


Language.factories["bert_wordpiecer"] = BertWordPiecer
