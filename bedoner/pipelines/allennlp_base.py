"""Basic class and utils for AllenNLP"""
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, Union

import torch
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.models.archival import load_archive
from spacy.pipeline import Pipe
from spacy.tokens import Doc

Pathlike = Union[str, Path]
VALIDATION = "validation"
VALIDATION_DATASET_READER = "validation_dataset_reader"
DATASET_READER = "dataset_reader"
ARCHIVE = "archive"


class AllennlpPipe(Pipe):
    def __init__(
        self,
        model: Model = None,
        dataset_reader: DatasetReader = None,
        archive_path: Path = None,
        **cfg
    ):
        self.model = model
        self.dataset_reader = dataset_reader
        self.cfg = cfg
        self.archive_path = archive_path

    @classmethod
    def from_archive(
        cls, archive_path: Pathlike, dataset_reader_to_load: str = VALIDATION
    ):
        archive = load_archive(str(archive_path))
        config = archive.config
        if dataset_reader_to_load == VALIDATION and VALIDATION_DATASET_READER in config:
            dataset_reader_params = config[VALIDATION_DATASET_READER]
        else:
            dataset_reader_params = config[DATASET_READER]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)
        return cls(
            model=archive.model,
            dataset_reader=dataset_reader,
            config={"allen_archive": archive.config},
            archive_path=Path(archive_path).absolute(),
        )

    def to_disk(self, path: Pathlike, exclude=tuple(), **kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True)
        if self.archive_path:
            shutil.copytree(self.archive_path, path / ARCHIVE)

        with (path / "cfg.pkl").open("wb") as f:
            pickle.dump(self.cfg, f)

    def from_disk(self, path: Pathlike, exclude=tuple(), **kwargs) -> "AllennlpPipe":
        path = Path(path)
        archive_path = path / ARCHIVE
        new_self = self.from_archive(archive_path)
        with (path / "cfg.pkl").open("rb") as f:
            cfg = pickle.load(f)
        self.model = new_self.model
        self.dataset_reader = new_self.dataset_reader
        self.cfg.update(cfg)
        self.archive_path = archive_path
        return self

    def predict(self, docs: Iterable[Doc]) -> Dict:
        self.model.eval()
        with torch.no_grad():
            tokens_list = [[t.text for t in doc] for doc in docs]
            instances = [
                self.dataset_reader.text_to_instance(tokens) for tokens in tokens_list
            ]
            outputs = self.model.forward_on_instances(instances)
        return outputs
