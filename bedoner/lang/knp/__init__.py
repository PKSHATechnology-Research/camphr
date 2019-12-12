import warnings

warnings.warn(
    "knp.Japanese has been deprecated. Please use models.knp.", DeprecationWarning
)


class Japanese:
    def __init__(self):
        warnings.warn(
            "knp.Japanese has been deprecated. Please use models.knp.",
            DeprecationWarning,
        )


class TorchJapanese(Japanese):
    lang = "torch_knp"
