"""Module trf_model defines pytorch-transformers components."""
import dataclasses
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast

import torch
import transformers as trf
from spacy.gold import GoldParse
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
from spacy_transformers.util import ATTRS


from bedoner.torch_utils import (
    OptimizerParameters,
    TensorWrapper,
    TorchPipe,
    get_parameters_with_decay,
)
from bedoner.utils import zero_pad

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-ja-juman": "s3://bedoner/trf_models/bert/bert-ja-juman.bin"
}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-ja-juman": "s3://bedoner/trf_models/bert/bert-ja-juman-config.json"
}


@dataclasses.dataclass
class BertModelInputs:
    """Container for BERT model input. See `trf.BertModel`'s docstring for detail."""

    input_ids: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    head_mask: Optional[torch.Tensor] = None


@dataclasses.dataclass
class BertModelOutputs:
    """A container for BertModel outputs. See `trf.BertModel`'s docstring for detail."""

    laste_hidden_state: torch.FloatTensor  # shape ``(batch_size, sequence_length, hidden_size)``
    pooler_output: torch.FloatTensor  # shape ``(batch_size, hidden_size)``
    hidden_states: Optional[torch.FloatTensor] = None
    # list of (one for the output of each layer + the output of the embeddings) of shape ``(batch_size, sequence_length, hidden_size)``
    attensions: Optional[torch.FloatTensor] = None
    # list of (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``


class BertModel(TorchPipe):
    """Pytorch transformers BertModel component.

    Attach BERT outputs to doc.
    """

    name = "bert"
    trf_model_cls = trf.BertModel
    trf_config_cls = trf.BertConfig

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
        cfg["trf_name"] = name_or_path
        model = cls.Model(from_pretrained=True, **cfg)
        cfg["trf_config"] = dict(model.config.to_dict())
        return cls(vocab, model=model, **cfg)

    @classmethod
    def Model(cls, **cfg) -> trf.BertModel:
        """Create `trf.BertModel`"""
        if cfg.get("from_pretrained"):
            cls.trf_model_cls.pretrained_model_archive_map.update(
                BERT_PRETRAINED_MODEL_ARCHIVE_MAP
            )
            cls.trf_model_cls.config_class.pretrained_config_archive_map.update(
                BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
            )
            model = cls.trf_model_cls.from_pretrained(cfg.get("trf_name"))
        else:
            if "vocab_size" in cfg["trf_config"]:
                vocab_size = cfg["trf_config"]["vocab_size"]
                cfg["trf_config"]["vocab_size_or_config_json_file"] = vocab_size
            model = cls.trf_model_cls(cls.trf_config_cls(**cfg["trf_config"]))
        return model

    @property
    def max_length(self) -> int:
        return self.model.config.max_position_embeddings

    def assert_length(self, x: BertModelInputs):
        if x.input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Too long input_ids. Expected {self.max_length}, but got {x.input_ids.shape[1]}"
            )

    def predict(self, docs: List[Doc]) -> BertModelOutputs:
        self.require_model()
        self.model.eval()
        x = self.docs_to_trfinput(docs)
        self.assert_length(x)
        with torch.no_grad():
            y = BertModelOutputs(*self.model(**dataclasses.asdict(x)))
        return y

    def set_annotations(self, docs: List[Doc], outputs: BertModelOutputs) -> None:
        """Assign the extracted features to the Doc."""
        for i, doc in enumerate(docs):
            length = len(doc._.trf_word_pieces)
            # Instead of assigning tensor directory, assign `TensorWrapper`
            # so that trailing pipe can handle batch tensor efficiently.
            doc._.trf_last_hidden_state = TensorWrapper(
                outputs.laste_hidden_state, i, length
            )
            doc._.trf_pooler_output = TensorWrapper(outputs.pooler_output, i, length)
            lh: torch.Tensor = doc._.get(ATTRS.last_hidden_state).get()

            doc_tensor = lh.new_zeros((len(doc), lh.shape[-1]))
            # TODO: Inefficient
            # TODO: Store the functionality into user_hooks after https://github.com/explosion/spaCy/issues/4439 was released
            for i, a in enumerate(doc._.get(ATTRS.alignment)):
                doc_tensor[i] += lh[a].sum(0)
            doc.tensor = doc_tensor
            doc.user_hooks["vector"] = get_doc_vector_via_tensor
            doc.user_span_hooks["vector"] = get_span_vector_via_tensor
            doc.user_token_hooks["vector"] = get_token_vector_via_tensor
            doc.user_hooks["similarity"] = get_similarity
            doc.user_span_hooks["similarity"] = get_similarity
            doc.user_token_hooks["similarity"] = get_similarity

            if outputs.hidden_states:
                doc._.trf_all_hidden_states = [
                    TensorWrapper(hid_layer, i)
                    for hid_layer in cast(Iterable, outputs.hidden_states)
                ]
            if outputs.attensions:
                doc._.trf_all_attentions = [
                    TensorWrapper(attention_layer, i)
                    for attention_layer in cast(Iterable, outputs.attensions)
                ]

    def update(self, docs: List[Doc], golds: List[GoldParse]):
        """Simply forward docs in training mode."""
        self.require_model()
        self.model.train()
        x = self.docs_to_trfinput(docs)
        self.assert_length(x)
        y = BertModelOutputs(*self.model(**dataclasses.asdict(x)))
        self.set_annotations(docs, y)

    def docs_to_trfinput(self, docs: List[Doc]) -> BertModelInputs:
        """Generate input data for trf model from docs."""
        inputs = BertModelInputs(
            input_ids=torch.tensor(
                zero_pad([doc._.trf_word_pieces for doc in docs]), device=self.device
            )
        )
        inputs.attention_mask = torch.tensor(
            zero_pad([[1 for _ in range(len(doc._.trf_word_pieces))] for doc in docs]),
            device=self.device,
        )

        segments = []
        for doc in docs:
            seg = []
            for i, s in enumerate(doc._.trf_segments):
                seg += [i] * len(s._.trf_word_pieces)
            segments.append(seg)
        inputs.token_type_ids = torch.tensor(zero_pad(segments), device=self.device)
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

    def from_disk(self, path: Path, exclude=tuple(), **kwargs) -> "BertModel":
        with (path / "cfg.pkl").open("rb") as f:
            self.cfg = pickle.load(f)
        with (path / "vocab.pkl").open("rb") as f:
            self.vocab = pickle.load(f)
        self.model = self.trf_model_cls.from_pretrained(path)
        return self


def get_doc_vector_via_tensor(doc) -> torch.Tensor:
    return doc.tensor.sum(0)


def get_span_vector_via_tensor(span) -> torch.Tensor:
    return span.doc.tensor[span.start : span.end].sum(0)


def get_token_vector_via_tensor(token) -> torch.Tensor:
    return token.doc.tensor[token.i]


def get_similarity(o1: Union[Doc, Span, Token], o2: Union[Doc, Span, Token]) -> int:
    v1: torch.Tensor = o1.vector
    v2: torch.Tensor = o2.vector
    return (v1.dot(v2) / (v1.norm() * v2.norm())).item()


Language.factories[BertModel.name] = BertModel
