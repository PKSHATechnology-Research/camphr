# Named entity recognition pipe component for transformers
from typing import Any, Dict, TypedDict, List, cast, ClassVar
from dataclasses import dataclass, asdict
from camphr.serde import SerDe, SerDeDataclassMixin
from pathlib import Path
from camphr.doc import Doc, DocProto, Ent
from transformers.models.auto import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines.token_classification import TokenClassificationPipeline
from transformers.pipelines import pipeline as trf_pipeline
from camphr.nlp import Nlp


class TrfEnt(TypedDict):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


class Ner(Nlp, SerDe):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer_kwargs: Dict[str, Any] = {},
        model_kwargs: Dict[str, Any] = {},
    ):
        self.pipeline = self._create_pipeline(
            pretrained_model_name_or_path, tokenizer_kwargs, model_kwargs
        )
        self.tokenizer_kwargs = tokenizer_kwargs
        self.model_kwargs = model_kwargs

    @staticmethod
    def _create_pipeline(
        pretrained_model_name_or_path: str,
        tokenizer_kwargs: Dict[str, Any],
        model_kwargs: Dict[str, Any],
    ) -> "TokenClassificationPipeline":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )
        pipeline = trf_pipeline(  # type: ignore
            "ner", model=model, tokenizer=tokenizer  # type: ignore
        )
        if not isinstance(pipeline, TokenClassificationPipeline):
            raise ValueError(
                f"Internal Error: expected `TokenClassificationPipeline`, got {type(pipeline)}"
            )
        return pipeline

    def __call__(self, text: str) -> DocProto:
        ents_raw = self.pipeline(text)
        trf_ents = cast(List[TrfEnt], self.pipeline.group_entities(ents_raw))
        doc = Doc(text)
        doc.ents = []
        for ent in trf_ents:
            e = Ent(
                ent["start"],
                ent["end"],
                doc,
                label=ent["entity_group"],
                score=ent["score"],
            )
            doc.ents.append(e)
        return doc

    # ser/de
    @dataclass
    class _SerdeMeta(SerDeDataclassMixin):
        tokenizer_kwargs: Dict[str, Any]
        model_kwargs: Dict[str, Any]
        FILENAME: ClassVar[str] = "camphr_meta.json"

    def to_disk(self, path: Path):
        meta = self._SerdeMeta(self.tokenizer_kwargs, self.model_kwargs)
        meta.to_disk(path)
        self.pipeline.save_pretrained(str(path))

    @classmethod
    def from_disk(cls, path: Path) -> "Ner":
        meta = cls._SerdeMeta.from_disk(path)
        return cls(str(path), **asdict(meta))
