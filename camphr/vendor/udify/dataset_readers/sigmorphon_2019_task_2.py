import logging
import re
from typing import Any, Callable, Dict, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from .lemma_edit import gen_lemma_rule
from .parser import DEFAULT_FIELDS, parse_line

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# A dictionary version of the UniMorph schema specified at https://unimorph.github.io/doc/unimorph-schema.pdf
unimorph_schema = {
    "aktionsart": [
        "accmp",
        "ach",
        "acty",
        "atel",
        "dur",
        "dyn",
        "pct",
        "semel",
        "stat",
        "tel",
    ],
    "animacy": ["anim", "hum", "inan", "nhum"],
    "argument_marking": ["argac3s"],
    "aspect": ["hab", "ipfv", "iter", "pfv", "prf", "prog", "prosp"],
    "case": [
        "abl",
        "abs",
        "acc",
        "all",
        "ante",
        "apprx",
        "apud",
        "at",
        "avr",
        "ben",
        "byway",
        "circ",
        "com",
        "compv",
        "dat",
        "eqtv",
        "erg",
        "ess",
        "frml",
        "gen",
        "in",
        "ins",
        "inter",
        "nom",
        "noms",
        "on",
        "onhr",
        "onvr",
        "post",
        "priv",
        "prol",
        "propr",
        "prox",
        "prp",
        "prt",
        "rel",
        "rem",
        "sub",
        "term",
        "trans",
        "vers",
        "voc",
    ],
    "comparison": ["ab", "cmpr", "eqt", "rl", "sprl"],
    "definiteness": ["def", "indf", "nspec", "spec"],
    "deixis": [
        "abv",
        "bel",
        "even",
        "med",
        "noref",
        "nvis",
        "phor",
        "prox",
        "ref1",
        "ref2",
        "remt",
        "vis",
    ],
    "evidentiality": [
        "assum",
        "aud",
        "drct",
        "fh",
        "hrsy",
        "infer",
        "nfh",
        "nvsen",
        "quot",
        "rprt",
        "sen",
    ],
    "finiteness": ["fin", "nfin"],
    "gender": ["bantu1-23", "fem", "masc", "nakh1-8", "neut"],
    "information_structure": ["foc", "top"],
    "interrogativity": ["decl", "int"],
    "language-specific_features": ["lgspec1", "lgspec2"],
    "mood": [
        "adm",
        "aunprp",
        "auprp",
        "cond",
        "deb",
        "ded",
        "imp",
        "ind",
        "inten",
        "irr",
        "lkly",
        "oblig",
        "opt",
        "perm",
        "pot",
        "purp",
        "real",
        "sbjv",
        "sim",
    ],
    "number": ["du", "gpauc", "grpl", "invn", "pauc", "pl", "sg", "tri"],
    "part_of_speech": [
        "adj",
        "adp",
        "adv",
        "art",
        "aux",
        "clf",
        "comp",
        "conj",
        "det",
        "intj",
        "n",
        "num",
        "part",
        "pro",
        "propn",
        "v",
        "v.cvb",
        "v.msdr",
        "v.ptcp",
    ],
    "person": ["0", "1", "2", "3", "4", "excl", "incl", "obv", "prx"],
    "polarity": ["pos", "neg"],
    "politeness": [
        "avoid",
        "col",
        "elev",
        "foreg",
        "form",
        "high",
        "humb",
        "infm",
        "lit",
        "low",
        "pol",
        "stelev",
        "stsupr",
    ],
    "possession": [
        "aln",
        "naln",
        "pss1d",
        "pss1de",
        "pss1di",
        "pss1p",
        "pss1pe",
        "pss1pi",
        "pss1s",
        "pss2d",
        "pss2df",
        "pss2dm",
        "pss2p",
        "pss2pf",
        "pss2pm",
        "pss2s",
        "pss2sf",
        "pss2sform",
        "pss2sinfm",
        "pss2sm",
        "pss3d",
        "pss3df",
        "pss3dm",
        "pss3p",
        "pss3pf",
        "pss3pm",
        "pss3s",
        "pss3sf",
        "pss3sm",
        "pssd",
        "psss",
        "pssp",
        "pss",
        "pss3",
        "pss1",
    ],
    "switch-reference": [
        "mn",
        "ds",
        "dsadv",
        "log",
        "or",
        "seqma",
        "simma",
        "ss",
        "ssadv",
    ],
    "tense": ["1day", "fut", "hod", "immed", "prs", "pst", "rct", "rmt"],
    "valency": ["appl", "caus", "ditr", "imprs", "intr", "recp", "refl", "tr"],
    "voice": [
        "acfoc",
        "act",
        "agfoc",
        "antip",
        "bfoc",
        "cfoc",
        "dir",
        "ifoc",
        "inv",
        "lfoc",
        "mid",
        "pass",
        "pfoc",
    ]
    # Extra Undefined Labels: "arg*", "pss*", "dist", "prontype", "number", "mood"
}


def lazy_parse(text: str, fields: Tuple[str, ...] = DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            # TODO: upgrade conllu library
            yield [
                parse_line(line, fields, parse_feats=False)
                for line in sentence.split("\n")
                if line and not line.strip().startswith("#")
            ]


@DatasetReader.register("udify_sigmorphon_2019_task_2")
class Sigmorphon2019Task2DatasetReader(DatasetReader):
    def __init__(
        self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.label_to_dimension = {}
        for dimension, labels in unimorph_schema.items():
            for label in labels:
                self.label_to_dimension[label] = dimension

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
                feats = get_field("feats")

                yield self.text_to_instance(
                    words,
                    lemmas,
                    lemma_rules,
                    feats,
                    ids,
                    multiword_ids,
                    multiword_forms,
                )

    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        lemmas: List[str] = None,
        lemma_rules: List[str] = None,
        feats: List[str] = None,
        ids: List[str] = None,
        multiword_ids: List[str] = None,
        multiword_forms: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["tokens"] = tokens

        if lemma_rules:
            fields["lemmas"] = SequenceLabelField(
                lemma_rules, tokens, label_namespace="lemmas"
            )

        if "feats":
            fields["feats"] = SequenceLabelField(feats, tokens, label_namespace="feats")

            # TODO: parameter to turn this off
            feature_seq = []

            for feat in feats:
                features = feat.lower().split(";") if feat != "_" else "_"
                dimensions = {dimension: "_" for dimension in unimorph_schema}

                if feat != "_":
                    for label in features:
                        # Use regex to handle special cases where multi-labels are contained inside "{}"
                        first_label = re.findall(
                            r"(?#{)([a-zA-Z0-9.\-_]+)(?#\+|\/|})", label
                        )
                        first_label = first_label[0] if first_label else label

                        if first_label not in self.label_to_dimension:
                            if first_label.startswith("arg"):
                                # TODO: support argument_marking dimension
                                continue
                            elif first_label in ["dist", "prontype", "number", "mood"]:
                                # TODO: unknown labels
                                continue
                            elif first_label.startswith("pss"):
                                dimension = "possession"
                            else:
                                raise KeyError(first_label)
                        else:
                            dimension = self.label_to_dimension[first_label]

                        dimensions[dimension] = label

                feature_seq.append(dimensions)

            for dimension in unimorph_schema:
                labels = [f[dimension] for f in feature_seq]
                fields[dimension] = SequenceLabelField(
                    labels, tokens, label_namespace=dimension
                )

        fields["metadata"] = MetadataField(
            {
                "words": words,
                "feats": feats,
                "lemmas": lemmas,
                "lemma_rules": lemma_rules,
                "ids": ids,
                "multiword_ids": multiword_ids,
                "multiword_forms": multiword_forms,
            }
        )

        return Instance(fields)
