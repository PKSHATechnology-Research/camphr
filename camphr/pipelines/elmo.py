"""Elmo: Deep contextualized word representations (https://allennlp.org/elmo)"""
import json
import shutil
from pathlib import Path
from typing import Generator, Iterable, List

import numpy as np
import spacy
import torch
from spacy.pipeline import Pipe
from spacy.tokens import Doc

from camphr.pipelines.allennlp_base import Pathlike
from camphr.pipelines.utils import (
    get_doc_vector_via_tensor,
    get_similarity,
    get_span_vector_via_tensor,
    get_token_vector_via_tensor,
)


@spacy.component("elmo", assigns=["doc.tensor", "doc.vector", "token.vector"])
class Elmo(Pipe):
    WEIGHTS_FILE_NAME = "weights.hdf5"
    OPTIONS_FILE_NAME = "options.json"

    def __init__(self, model=None, **cfg):
        self.model = model
        self.cfg = cfg

    @classmethod
    def from_elmofiles(cls, options_file: Pathlike, weight_file: Pathlike, **cfg):
        """Construct Elmo with elmo files.
        
        Note:
            Elmo files are distributed in https://allennlp.org/elmo
        
        Example:
            >>> elmo = Elmo.from_elmofiles("./options.json", "./weights.hd5")
            >>> nlp = spacy.blank("en")
            >>> nlp.add_pipe(elmo)
        """
        model = cls.Model(options_file, weight_file)
        return cls(model, **cfg)

    @classmethod
    def Model(cls, options_file: Pathlike, weight_file: Pathlike, **cfg):
        from allennlp.modules.elmo import Elmo

        return Elmo(
            str(Path(options_file).absolute()), str(Path(weight_file).absolute()), 3
        )

    def to_disk(self, path: Pathlike, **cfg):
        path = Path(path)
        path.mkdir(exist_ok=True)
        shutil.copy(
            self.model._elmo_lstm._token_embedder._weight_file,
            str(path / self.WEIGHTS_FILE_NAME),
        )
        with (path / self.OPTIONS_FILE_NAME).open("w") as f:
            json.dump(self.model._elmo_lstm._token_embedder._options, f)

    def from_disk(self, path: Pathlike, **cfg):
        path = Path(path)
        self.model = self.Model(
            path / self.OPTIONS_FILE_NAME, path / self.WEIGHTS_FILE_NAME
        )

    def predict(self, docs: List[Doc]) -> Generator[np.ndarray, None, None]:
        from allennlp.modules.elmo import batch_to_ids

        tokens = [[token.text for token in doc] for doc in docs]
        return self.model(batch_to_ids(tokens))["elmo_representations"]

    def set_annotations(self, docs: List[Doc], outputs: Iterable[torch.Tensor]):
        for doc, vec in zip(docs, outputs):
            vec = vec.detach().cpu().numpy()
            assert vec.shape[1] == len(doc), f"vector: {vec.shape}, doc: {len(doc)}"
            doc.tensor = vec[-1]
            doc.user_hooks["vector"] = get_doc_vector_via_tensor
            doc.user_span_hooks["vector"] = get_span_vector_via_tensor
            doc.user_token_hooks["vector"] = get_token_vector_via_tensor
            doc.user_hooks["similarity"] = get_similarity
            doc.user_span_hooks["similarity"] = get_similarity
            doc.user_token_hooks["similarity"] = get_similarity
            doc.user_data["elmo"] = vec
