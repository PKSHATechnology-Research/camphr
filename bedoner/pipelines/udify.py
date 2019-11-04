from typing import Dict, Iterable
from spacy.tokens import Doc, Token
from allennlp.common.util import import_submodules
from bedoner.vendor.udify.models.udify_model import OUTPUTS as UdifyOUTPUTS
from .allennlp_base import AllennlpPipe

import_submodules("bedoner.vendor.udify")


class Udify(AllennlpPipe):
    def set_annotations(self, docs: Iterable[Doc], outputs: Dict):
        for doc, output in zip(docs, outputs):
            deps = output[UdifyOUTPUTS.predicted_dependencies]
            heads = output[UdifyOUTPUTS.predicted_heads]
            uposes = output[UdifyOUTPUTS.upos]
            lemmas = output[UdifyOUTPUTS.lemmas]
            words = output[UdifyOUTPUTS.words]
            _doc_tokens = [token.text for token in doc]
            if not words == [token.text for token in doc]:
                raise ValueError(
                    "Internal error has occured."
                    f"Input text: {doc.text}\n"
                    f"Input tokens: {_doc_tokens}\n"
                    f"Model words: {words}"
                )
            if not (
                len(deps) == len(heads) == len(uposes) == len(words) == len(lemmas)
            ):
                raise ValueError(
                    "Internal error has occured.\n"
                    f"text: {doc.text}\n"
                    f"deps: {deps}\n"
                    f"heads: {heads}\n"
                    f"uposes: {uposes}\n"
                    f"words: {words}\n"
                    f"lemmas: {lemmas}\n"
                )

            for token, dep, head, upos, lemma in zip(doc, deps, heads, uposes, lemmas):
                token: Token = token
                token.dep_ = dep
                token.lemma_ = lemma
                token.pos_ = upos
                token.head = doc[head]
            doc.is_parsed = True
