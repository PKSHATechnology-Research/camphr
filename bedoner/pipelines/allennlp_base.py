"""Basic class and utils for AllenNLP"""
from pathlib import Path
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.data import DatasetReader
from typing import Dict, Iterable, Union
from spacy.tokens import Doc

Pathlike = Union[str, Path]
VALIDATION = "validation"
VALIDATION_DATASET_READER = "validation_dataset_reader"
DATASET_READER = "dataset_reader"


class AllennlpPipe:
    def __init__(
        self, model: Model = None, dataset_reader: DatasetReader = None, **cfg
    ):
        self.model = model
        self.dataset_reader = dataset_reader
        self.cfg = cfg

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
        )

    def predict(self, docs: Iterable[Doc]) -> Dict:
        tokens_list = [[t.text for t in doc] for doc in docs]
        instances = [
            self.dataset_reader.text_to_instance(tokens) for tokens in tokens_list
        ]
        outputs = self.model.forward_on_instances(instances)
        return outputs
