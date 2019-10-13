"""Module trf_ner defines pytorch transformers NER model

Models defined in this modules must be used with `bedoner.pipelines.trf_model`'s model in `spacy.Language` pipeline
"""
import pickle
from pathlib import Path
from typing import Iterable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as trf
from spacy.gold import GoldParse, spans_from_biluo_tags
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
from transformers.modeling_bert import BertConfig

from bedoner.pipelines.trf_model import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertModel
from bedoner.pipelines.utils import UNK, correct_biluo_tags
from bedoner.torch_utils import (
    OptimizerParameters,
    TensorWrapper,
    TorchPipe,
    get_parameters_with_decay,
)


class BertTokenClassifier(nn.Module):
    """A thin layer to classifier"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, labels=None
    ) -> torch.Tensor:

        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


class BertForTokenClassification(TorchPipe):
    """Base class for token classification task (e.g. NER).

    Requires `BertModel` before this model in the pipeline to use this model.

    Notes:
        `Token._.cls_logit` is set and stored the output of this model into. This is usefule to calculate the probability of the classification.
        `Doc._.cls_logit` is set and stored the output of this model into.
    """

    name = "bert_tokenclassifier"
    trf_model_cls = BertTokenClassifier
    trf_config_cls = trf.BertConfig

    def __init__(self, vocab, model=True, **cfg):
        self.vocab = vocab
        self.model = model
        self.cfg = cfg

    @staticmethod
    def install_extensions():
        token_exts = ["cls_logit"]
        doc_exts = ["cls_logit"]
        for ext in token_exts:
            if Token.get_extension(ext) is None:
                Token.set_extension(ext, default=None)
        for ext in doc_exts:
            if Doc.get_extension(ext) is None:
                Doc.set_extension(ext, default=None)

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    @classmethod
    def from_pretrained(cls, vocab: Vocab, name: str, **cfg):
        cfg["trf_name"] = name
        model = cls.Model(from_pretrained=True, **cfg)
        cfg["trf_config"] = dict(model.config.to_dict())
        return cls(vocab, model=model, **cfg)

    @classmethod
    def Model(cls, **cfg) -> BertTokenClassifier:
        assert cfg.get("labels")
        cfg.setdefault("trf_config", {})
        cfg["trf_config"]["num_labels"] = len(cfg.get("labels", []))
        if cfg.get("from_pretrained"):
            cls.trf_config_cls.pretrained_config_archive_map.update(
                BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
            )
            config = cls.trf_config_cls.from_pretrained(
                cfg["trf_name"], **cfg["trf_config"]
            )
            model = BertTokenClassifier(config)
        else:
            if "vocab_size" in cfg["trf_config"]:
                vocab_size = cfg["trf_config"]["vocab_size"]
                cfg["trf_config"]["vocab_size_or_config_json_file"] = vocab_size
            model = cls.BertClassifier(trf.BertConfig(**cfg["trf_config"]))
        assert model.config.num_labels == len(cfg["labels"])
        return model

    @property
    def labels(self):
        return tuple(self.cfg.setdefault("labels", []))

    @property
    def label2id(self):
        return {v: i for i, v in enumerate(self.labels)}

    def predict(self, docs: Iterable[Doc]) -> torch.Tensor:
        self.require_model()
        self.model.eval()
        with torch.no_grad():
            x: TensorWrapper = next(
                iter(docs)
            )._.trf_last_hidden_state  # assumed that the batch tensor of all docs is stored into the extension.
            logits = self.model(x.batch_tensor)
        return logits

    def update(self, docs: Iterable[Doc], golds: Iterable[GoldParse]):
        raise NotImplementedError

    def set_annotations(
        self, docs: Iterable[Doc], logits: torch.Tensor
    ) -> Iterable[Doc]:
        raise NotImplementedError

    def optim_parameters(self) -> OptimizerParameters:
        no_decay = self.cfg.get("no_decay")
        weight_decay = self.cfg.get("weight_decay")
        return get_parameters_with_decay(self.model, no_decay, weight_decay)

    def to_disk(self, path: Path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        model: BertTokenClassifier = self.model
        model.config.save_pretrained(path)
        torch.save(model.state_dict(), str(path / "model.pth"))

        with (path / "cfg.pkl").open("wb") as f:
            pickle.dump(self.cfg, f)

        # TODO: This may not be good way because vocab is saved separetely.
        with (path / "vocab.pkl").open("wb") as f:
            pickle.dump(self.vocab, f)

    def from_disk(self, path: Path, exclude=tuple(), **kwargs) -> BertModel:
        config = self.trf_config_cls.from_pretrained(path)
        model = BertTokenClassifier(config)
        model.load_state_dict(
            torch.load(str(path / "model.pth"), map_location=self.device)
        )
        model.eval()
        self.model = model

        with (path / "cfg.pkl").open("rb") as f:
            self.cfg = pickle.load(f)
        with (path / "vocab.pkl").open("rb") as f:
            self.vocab = pickle.load(f)
        return self


class BertForNamedEntityRecognition(BertForTokenClassification):
    """Named entity recognition component with pytorch-transformers."""

    name = "bert_ner"
    trf_model_cls = BertTokenClassifier
    trf_config_cls = trf.BertConfig

    @property
    def ignore_label_index(self) -> int:
        if UNK in self.labels:
            return self.labels.index(UNK)
        return -1

    def update(self, docs: Iterable[Doc], golds: Iterable[GoldParse]):
        self.require_model()
        label2id = self.label2id
        ignore_index = self.ignore_label_index
        # TODO: Batch
        x: TensorWrapper = next(iter(docs))._.trf_last_hidden_state
        logits = self.model(x.batch_tensor)
        for doc, gold, logit in zip(docs, golds, logits):
            # use first wordpiece for each tokens
            try:
                idx = list(map(lambda x: x[0], doc._.trf_alignment))
            except RuntimeError as e:
                raise ValueError(
                    f"Internal error is occured when processing '{doc}'. This seems to be an error about wordpiecer mapping, please notify maintainer.\n"
                    + e.message
                )
            loss = F.cross_entropy(
                logit[idx],
                torch.tensor([label2id[ner] for ner in gold.ner], device=self.device),
                ignore_index=ignore_index,
            )

            doc._.cls_logit = logit
            doc._.loss = loss

    def set_annotations(
        self, docs: Iterable[Doc], logits: torch.Tensor
    ) -> Iterable[Doc]:
        """Modify a batch of documents, using pre-computed scores."""
        assert len(logits.shape) == 3  # (batch, length, nclass)
        id2label = self.labels

        for doc, logit in zip(docs, cast(Iterable, logits)):
            ids = torch.argmax(logit, dim=1)
            labels = [id2label[r] for r in cast(Iterable, ids)]
            doc._.cls_logit = logit
            biluo_tags = []
            for token, a in zip(doc, doc._.trf_alignment):
                token._.cls_logit = logit[a[0]]
                label = labels[a[0]]
                biluo_tags.append(label)
            biluo_tags = correct_biluo_tags(biluo_tags)
            doc.ents = spans_from_biluo_tags(doc, biluo_tags)
        return docs


BertForTokenClassification.install_extensions()
Language.factories[BertForTokenClassification.name] = BertForTokenClassification
Language.factories[BertForNamedEntityRecognition.name] = BertForNamedEntityRecognition
