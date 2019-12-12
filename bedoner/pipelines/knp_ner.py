import warnings

NEW_MODULE = "bedoner.pipelines.knp"
warnings.warn(
    f"knp_ner has been deprecated. Please use {NEW_MODULE}.", DeprecationWarning
)


class KnpEntityExtractor:
    def __init__(self) -> None:
        warnings.warn(
            f"KnpEntityExtractor has been deprecated. Please use {NEW_MODULE}.KNP.",
            DeprecationWarning,
        )
