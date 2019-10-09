"""Module pytt_model defines pytorch-transformers components."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import dataclasses
import transformers as trf
import torch
from bedoner.torch_utils import (
    OptimizerParameters,
    TensorWrapper,
    TorchPipe,
    get_parameters_with_decay,
)
from bedoner.utils import zero_pad
from spacy.gold import GoldParse
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-ja-juman": "s3://bedoner/pytt_models/bert/bert-ja-juman.bin"
}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-ja-juman": "s3://bedoner/pytt_models/bert/bert-ja-juman-config.json"
}


class BertModel(TorchPipe):
    """Pytorch transformers BertModel component.

    Attach BERT outputs to doc.
    """

    name = "pytt_bert"
    pytt_model_cls = trf.BertModel
    pytt_config_cls = trf.BertConfig

    def __init__(self, vocab, model=True, **cfg):
        self.vocab = vocab
        self.model = model
        self.cfg = cfg

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        return cls(nlp.vocab, **cfg)

    @classmethod
    def from_pretrained(cls, vocab: Vocab, name_or_path: str, **cfg):
        """Load pretrained model."""
        cfg["pytt_name"] = name_or_path
        model = cls.Model(from_pretrained=True, **cfg)
        cfg["pytt_config"] = dict(model.config.to_dict())
        return cls(vocab, model=model, **cfg)

    @classmethod
    def Model(cls, **cfg) -> trf.BertModel:
        """Create `pytt.BertModel`"""
        if cfg.get("from_pretrained"):
            cls.pytt_model_cls.pretrained_model_archive_map.update(
                BERT_PRETRAINED_MODEL_ARCHIVE_MAP
            )
            cls.pytt_model_cls.config_class.pretrained_config_archive_map.update(
                BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
            )
            model = cls.pytt_model_cls.from_pretrained(cfg.get("pytt_name"))
        else:
            if "vocab_size" in cfg["pytt_config"]:
                vocab_size = cfg["pytt_config"]["vocab_size"]
                cfg["pytt_config"]["vocab_size_or_config_json_file"] = vocab_size
            model = cls.pytt_model_cls(cls.pytt_config_cls(**cfg["pytt_config"]))
        return model

    def predict(self, docs: List[Doc]) -> BertModelOutputs:
        self.require_model()
        self.model.eval()
        x = self.docs_to_pyttinput(docs)
        with torch.no_grad():
            y = BertModelOutputs(*self.model(**dataclasses.asdict(x)))
        return y

    def set_annotations(self, docs: List[Doc], outputs: BertModelOutputs) -> None:
        """Assign the extracted features to the Doc."""
        for i, doc in enumerate(docs):
            length = len(doc._.pytt_word_pieces)
            # Instead of assigning tensor directory, assign `TensorWrapper`
            # so that trailing pipe can handle batch tensor efficiently.
            doc._.pytt_last_hidden_state = TensorWrapper(
                outputs.laste_hidden_state, i, length
            )
            doc._.pytt_pooler_output = TensorWrapper(outputs.pooler_output, i, length)
            if outputs.hidden_states:
                doc._.pytt_all_hidden_states = [
                    TensorWrapper(hid_layer, i) for hid_layer in outputs.hidden_states
                ]
            if outputs.attensions:
                doc._.pytt_all_attentions = [
                    TensorWrapper(attention_layer, i)
                    for attention_layer in outputs.attensions
                ]

    def update(self, docs: List[Doc], golds: List[GoldParse]):
        """Simply forward docs in training mode."""
        self.require_model()
        self.model.train()
        x = self.docs_to_pyttinput(docs)
        y = BertModelOutputs(*self.model(**dataclasses.asdict(x)))
        self.set_annotations(docs, y)

    def docs_to_pyttinput(self, docs: List[Doc]) -> BertModelInputs:
        """Generate input data for pytt model from docs."""
        inputs = BertModelInputs(
            input_ids=torch.tensor(zero_pad([doc._.pytt_word_pieces for doc in docs]))
        )
        inputs.attention_mask = torch.tensor(
            zero_pad([[1 for _ in range(len(doc._.pytt_word_pieces))] for doc in docs])
        )

        segments = []
        for doc in docs:
            seg = []
            for i, s in enumerate(doc._.pytt_segments):
                seg += [i] * len(s._.pytt_word_pieces)
            segments.append((seg))
        inputs.token_type_ids = torch.tensor(zero_pad(segments))
        return inputs

    def optim_parameters(self) -> OptimizerParameters:
        no_decay = self.cfg.get("no_decay")
        weight_decay = self.cfg.get("weight_decay")
        return get_parameters_with_decay(self.model, no_decay, weight_decay)

    def to_disk(self, path: Path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        model: trf.PreTrainedModel = self.model
        model.save_pretrained(str(path))
        with (path / "cfg.pkl").open("wb") as f:
            pickle.dump(self.cfg, f)

        # TODO: This may not be good because vocab is saved separetely.
        with (path / "vocab.pkl").open("wb") as f:
            pickle.dump(self.vocab, f)

    def from_disk(self, path: Path, exclude=tuple(), **kwargs) -> PyttBertModel:
        with (path / "cfg.pkl").open("rb") as f:
            self.cfg = pickle.load(f)
        with (path / "vocab.pkl").open("rb") as f:
            self.vocab = pickle.load(f)
        self.model = self.pytt_model_cls.from_pretrained(path)
        return self


@dataclasses.dataclass
class BertModelInputs:
    """Container for BERT model input. See `pytt.BertModel`'s docstring for detail."""

    input_ids: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    head_mask: Optional[torch.Tensor] = None


@dataclasses.dataclass
class BertModelOutputs:
    """A container for BertModel outputs. See `pytt.BertModel`'s docstring for detail."""

    laste_hidden_state: torch.FloatTensor  # shape ``(batch_size, sequence_length, hidden_size)``
    pooler_output: torch.FloatTensor  # shape ``(batch_size, hidden_size)``
    hidden_states: Optional[torch.FloatTensor] = None
    # list of (one for the output of each layer + the output of the embeddings) of shape ``(batch_size, sequence_length, hidden_size)``
    attensions: Optional[torch.FloatTensor] = None
    # list of (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``


Language.factories[PyttBertModel.name] = PyttBertModel
