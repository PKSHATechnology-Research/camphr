from typing import Dict, Iterable
from spacy.tokens import Doc
from allennlp.common.util import import_submodules
from .allennlp_base import AllennlpPipe

import_submodules("bedoner.vendor.udify")


class Udify(AllennlpPipe):
    def set_annotation(self, docs: Iterable[Doc], outputs: Dict):
        pass
