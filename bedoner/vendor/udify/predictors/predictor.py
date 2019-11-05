"""
The main UDify predictor to output conllu files
"""

from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("udify_predictor")
class UdifyPredictor(Predictor):
    """
    Predictor for a UDify model that takes in a sentence and returns
    a single set conllu annotations for it.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        if "@@UNKNOWN@@" not in self._model.vocab._token_to_index["lemmas"]:
            # Handle cases where the labels are present in the test set but not training set
            for instance in instances:
                self._predict_unknown(instance)
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        if "@@UNKNOWN@@" not in self._model.vocab._token_to_index["lemmas"]:
            # Handle cases where the labels are present in the test set but not training set
            self._predict_unknown(instance)
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def _predict_unknown(self, instance: Instance):
        """
        Maps each unknown label in each namespace to a default token
        :param instance: the instance containing a list of labels for each namespace
        """

        def replace_tokens(instance: Instance, namespace: str, token: str):
            if namespace not in instance.fields:
                return

            instance.fields[namespace].labels = [
                label
                if label in self._model.vocab._token_to_index[namespace]
                else token
                for label in instance.fields[namespace].labels
            ]

        replace_tokens(instance, "lemmas", "↓0;d¦")
        replace_tokens(instance, "feats", "_")
        replace_tokens(instance, "xpos", "_")
        replace_tokens(instance, "upos", "NOUN")
        replace_tokens(instance, "head_tags", "case")

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = sentence.split()
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        word_count = len([word for word in outputs["words"]])
        tags = [
            outputs[k] if k in outputs else ["_"] * word_count
            for k in [
                "ids",
                "words",
                "lemmas",
                "upos",
                "xpos",
                "feats",
                "predicted_heads",
                "predicted_dependencies",
            ]
        ]
        lines = zip(*tags)

        multiword_map = None
        if outputs["multiword_ids"]:
            multiword_ids = [
                [id] + [int(x) for x in id.split("-")]
                for id in outputs["multiword_ids"]
            ]
            multiword_forms = outputs["multiword_forms"]
            multiword_map = {
                start: (id_, form)
                for (id_, start, end), form in zip(multiword_ids, multiword_forms)
            }

        output_lines = []
        for i, line in enumerate(lines):
            line = [str(l) for l in line]

            # Handle multiword tokens
            if multiword_map and i + 1 in multiword_map:
                id_, form = multiword_map[i + 1]
                row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
                output_lines.append(row)

            row = "\t".join(line) + "".join(["\t_"] * 2)
            output_lines.append(row)

        return "\n".join(output_lines) + "\n\n"
