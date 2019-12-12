import warnings


class KnpEntityExtractor:
    def __init__(self) -> None:
        warnings.warn(
            "KnpEntityExtractor has been deprecated. Please use bedoner.pipelines.knp.KNP.",
            DeprecationWarning,
        )
