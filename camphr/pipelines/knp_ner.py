from ..errors import Warnings

NEW_MODULE = "camphr.pipelines.knp"
Warnings.W0("knp_ner", NEW_MODULE)


class KnpEntityExtractor:
    def __init__(self) -> None:
        Warnings.W0("KnpEntityExtractor", f"{NEW_MODULE}.KNP")
