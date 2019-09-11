from os.path import exists
from pathlib import Path
from spacy_pytorch_transformers.pipeline.wordpiecer import PyTT_WordPiecer
from spacy_pytorch_transformers._tokenizers import SerializableBertTokenizer
from spacy.language import Language
import shutil
import copy


class BertWordPiecer(PyTT_WordPiecer):
    name = "bert_wordpiecer"
    VOCAB_FILE = "vocab.txt"

    @classmethod
    def Model(cls, vocab_file: str, **kwargs) -> SerializableBertTokenizer:
        model = SerializableBertTokenizer(
            vocab_file=vocab_file, do_lower_case=False, tokenize_chinese_chars=False
        )  # do_lower_case=False: 濁点落ちを防ぐ，tokenize_chinese_chars=False: スペース以外のspiltを防ぐ
        return model

    def to_disk(self, path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        shutil.copy(self.cfg["vocab_file"], path / self.VOCAB_FILE)
        buf = copy.copy(self.cfg)
        del self.cfg["vocab_file"]
        super().to_disk(path, exclude, **kwargs)
        self.cfg = buf

    def from_disk(self, path, exclude=tuple(), **kwargs):
        self.cfg["vocab_file"] = str(path / self.VOCAB_FILE)
        super().from_disk(path, exclude, **kwargs)


Language.factories["bert_wordpiecer"] = BertWordPiecer
