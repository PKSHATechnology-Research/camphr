"""
A Dataset Reader for Universal Dependencies, with support for multiword tokens and special handling for NULL "_" tokens
"""

import logging
from typing import Any, Callable, Dict, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from .lemma_edit import gen_lemma_rule
from .parser import DEFAULT_FIELDS, parse_line

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def lazy_parse(text: str, fields: Tuple[str, ...] = DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            # TODO: upgrade conllu library
            yield [
                parse_line(line, fields)
                for line in sentence.split("\n")
                if line and not line.strip().startswith("#")
            ]


@DatasetReader.register("udify_universal_dependencies")
class UniversalDependenciesDatasetReader(DatasetReader):
    def __init__(
        self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in lazy_parse(conllu_file.read()):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
                annotation = [x for x in annotation if x["id"] is not None]

                if len(annotation) == 0:
                    continue

                def get_field(
                    tag: str, map_fn: Callable[[Any], Any] = None
                ) -> List[Any]:
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [
                        map_fn(x[tag]) if x[tag] is not None else "_"
                        for x in annotation
                        if tag in x
                    ]

                # Extract multiword token rows (not used for prediction, purely for evaluation)
                ids = [x["id"] for x in annotation]
                multiword_ids = [x["multi_id"] for x in multiword_tokens]
                multiword_forms = [x["form"] for x in multiword_tokens]

                words = get_field("form")
                lemmas = get_field("lemma")
                lemma_rules = [
                    gen_lemma_rule(word, lemma) if lemma != "_" else "_"
                    for word, lemma in zip(words, lemmas)
                ]
                upos_tags = get_field("upostag")
                xpos_tags = get_field("xpostag")
                feats = get_field(
                    "feats",
                    lambda x: "|".join(k + "=" + v for k, v in x.items())
                    if hasattr(x, "items")
                    else "_",
                )
                heads = get_field("head")
                dep_rels = get_field("deprel")
                dependencies = list(zip(dep_rels, heads))

                yield self.text_to_instance(
                    words,
                    lemmas,
                    lemma_rules,
                    upos_tags,
                    xpos_tags,
                    feats,
                    dependencies,
                    ids,
                    multiword_ids,
                    multiword_forms,
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        lemmas: List[str] = None,
        lemma_rules: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        feats: List[str] = None,
        dependencies: List[Tuple[str, int]] = None,
        ids: List[str] = None,
        multiword_ids: List[str] = None,
        multiword_forms: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["tokens"] = tokens

        names = ["upos", "xpos", "feats", "lemmas"]
        all_tags = [upos_tags, xpos_tags, feats, lemma_rules]
        for name, field in zip(names, all_tags):
            if field:
                fields[name] = SequenceLabelField(field, tokens, label_namespace=name)

        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies], tokens, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [int(x[1]) for x in dependencies],
                tokens,
                label_namespace="head_index_tags",
            )

        fields["metadata"] = MetadataField(
            {
                "words": words,
                "upos_tags": upos_tags,
                "xpos_tags": xpos_tags,
                "feats": feats,
                "lemmas": lemmas,
                "lemma_rules": lemma_rules,
                "ids": ids,
                "multiword_ids": multiword_ids,
                "multiword_forms": multiword_forms,
            }
        )

        return Instance(fields)
