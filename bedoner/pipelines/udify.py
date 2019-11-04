from typing import Dict, Iterable
from spacy.tokens import Doc, Token
from allennlp.common.util import import_submodules
from bedoner.vendor.udify.models.udify_model import OUTPUTS as UdifyOUTPUTS
from .allennlp_base import AllennlpPipe

import_submodules("bedoner.vendor.udify")


class Udify(AllennlpPipe):
    def set_annotation(self, docs: Iterable[Doc], outputs: Dict):
        for doc, output in zip(docs, outputs):
            deps = output[UdifyOUTPUTS.predicted_dependencies]
            heads = output[UdifyOUTPUTS.predicted_heads]
            uposes = output[UdifyOUTPUTS.upos]
            lemmas = output[UdifyOUTPUTS.lemmas]
            words = output[UdifyOUTPUTS.words]
            assert words == [token.text for token in doc]
            assert len(deps) == len(heads) == len(uposes) == len(words) == len(lemmas)
            for token, dep, head, lemma, upos in zip(doc, deps, heads, uposes, lemmas):
                token: Token = token
                token.dep_ = dep
                token.lemma_ = lemma
                token.pos_ = upos
                token.head = doc[head]
