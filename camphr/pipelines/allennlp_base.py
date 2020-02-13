"""Base class and utils for AllenNLP"""
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable

import torch
from spacy.pipeline import Pipe
from spacy.tokens import Doc

from camphr.pipelines.utils import flatten_docs_to_sents
from camphr.types import Pathlike

VALIDATION = "validation"
VALIDATION_DATASET_READER = "validation_dataset_reader"
DATASET_READER = "dataset_reader"
ARCHIVE = "archive"


class AllennlpPipe(Pipe):
    """Base class for allennlp Pipe.

    See .udify as an example.
    """

    def __init__(self, model=None, dataset_reader=None, archive_path=None, **cfg):
        self.model = model
        self.dataset_reader = dataset_reader
        self.cfg = cfg
        self.archive_path = archive_path

    @classmethod
    def from_archive(
        cls, archive_path: Pathlike, dataset_reader_to_load: str = VALIDATION
    ):
        """Construct from `allnlp.Archive`'s file."""
        # Uses lazy import because allennlp is an extra requirements.
        try:
            from allennlp.data import DatasetReader
            from allennlp.models.archival import load_archive
        except ImportError:
            raise ImportError(
                "Requires unofficial-allennlp-nightly.\nInstall it with `pip install unofficial-allennlp-nightly`."
            )

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

    def to_disk(self, path: Pathlike, **kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True)
        if self.archive_path:
            shutil.copytree(self.archive_path, path / ARCHIVE)

        with (path / "cfg.pkl").open("wb") as f:
            pickle.dump(self.cfg, f)

    def from_disk(self, path: Pathlike, **kwargs) -> "AllennlpPipe":
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
        all_sents = flatten_docs_to_sents(docs)
        with torch.no_grad():
            tokens_list = [[t.text for t in sent] for sent in all_sents]
            instances = [
                self.dataset_reader.text_to_instance(tokens) for tokens in tokens_list
            ]
            outputs = self.model.forward_on_instances(instances)
        return outputs
