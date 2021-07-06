# Named entity recognition pipe component for transformers
from typing import Any, Dict, TypedDict, List, cast
from camphr.pipe import Pipe
from camphr.serde import SerDe
from camphr.doc import Doc, DocProto, Ent, EntProto
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )
        pipeline = trf_pipeline(  # type: ignore
            "ner", model=self.model, tokenizer=self.tokenizer  # type: ignore
        )
        if not isinstance(pipeline, TokenClassificationPipeline):
            raise ValueError(
                f"Internal Error: expected `TokenClassificationPipeline`, got {type(pipeline)}"
            )
        self.pipeline = pipeline

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
