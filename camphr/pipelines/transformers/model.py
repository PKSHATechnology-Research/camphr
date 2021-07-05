"""Module trf_model defines pytorch-transformers components."""
from camphr.doc import Doc
import dataclasses
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import transformers

from camphr.pipelines.utils import get_similarity
from camphr.torch_utils import TensorWrapper, TorchPipe, set_grad

from .auto import get_trf_model_cls
from .tokenizer import TrfTokenizer
from .utils import ATTRS, TrfAutoMixin


@dataclasses.dataclass
class TrfModelInputs:
    input_ids: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


TRANSFORMERS_MODEL = "transformers_model"


class TrfModel(TrfAutoMixin[transformers.PreTrainedModel], TorchPipe):
    """Transformers Model component."""

    _TRF_NAME = "trf_name"
    _MODEL_CLS_GETTER = get_trf_model_cls

    @property
    def max_length(self) -> int:
        return self.model.config.max_position_embeddings

    def predict(self, docs: List[Doc]) -> torch.Tensor:
        self.model.eval()
        return self._apply_model(docs, False)

    def _apply_model(self, docs: List[Doc], grad: bool) -> torch.Tensor:
        self.require_model()
        x = TrfTokenizer.get_transformers_input(docs)
        assert x is not None
        x.to(device=self.device)
        with set_grad(grad):
            y = self.model(**x.model_input)
        return _get_last_hidden_state(y)

    def set_annotations(
        self, docs: List[Doc], outputs: torch.Tensor, set_vector: bool = True
    ) -> None:
        """Assign the extracted features to the Doc.

        Args:
            docs: List of `spacy.Doc`.
            outputs: Output from `self.predict`.
            set_vector: If True, attach the vector to doc. This may harms the performance.
        """
        for i, doc in enumerate(docs):
            length = len(doc._.get(ATTRS.tokens))
            # Instead of assigning a tensor directly, assign `TensorWrapper`
            # so that trailing pipes can handle batch tensors efficiently.
            doc._.set(ATTRS.last_hidden_state, TensorWrapper(outputs, i, length))

            if set_vector:
                lh: torch.Tensor = doc._.get(ATTRS.last_hidden_state).get()
                doc_tensor = lh.new_zeros((len(doc), lh.shape[-1]))
                # TODO: Inefficient
                # TODO: Store the functionality into user_hooks after https://github.com/explosion/spaCy/issues/4439 has been released
                for i, a in enumerate(doc._.get(ATTRS.align)):
                    if self.max_length > 0:
                        a = [aa for aa in a if aa < len(lh)]
                    doc_tensor[i] += lh[a].sum(0)
                doc.tensor = doc_tensor
                doc.user_hooks["vector"] = get_doc_vector_via_tensor
                doc.user_span_hooks["vector"] = get_span_vector_via_tensor
                doc.user_token_hooks["vector"] = get_token_vector_via_tensor
                doc.user_hooks["similarity"] = get_similarity
                doc.user_span_hooks["similarity"] = get_similarity
                doc.user_token_hooks["similarity"] = get_similarity

    @property
    def freeze(self) -> bool:
        if self.cfg.get("freeze"):
            return True
        return False

    def update(self, docs: List[Doc], golds: List[Any]):  # type: ignore
        """Simply forward `docs` in training mode."""
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()
        y = self._apply_model(docs, not self.freeze)
        # `set_vector = False` because the tensor may not be necessary in updating.
        # The tensor is still available via doc._.transformers_last_hidden_state.
        self.set_annotations(docs, y, set_vector=False)


def _get_last_hidden_state(output: Tuple[Any]) -> torch.Tensor:
    # assumes output[0] is the last hidden state
    return output[0]


def get_doc_vector_via_tensor(doc) -> np.ndarray:
    return doc.tensor.sum(0).cpu().numpy()


def get_span_vector_via_tensor(span) -> np.ndarray:
    return span.doc.tensor[span.start : span.end].sum(0).cpu().numpy()


def get_token_vector_via_tensor(token) -> np.ndarray:
    return token.doc.tensor[token.i].cpu().numpy()
