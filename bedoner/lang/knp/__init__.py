from ...errors import Warnings

Warnings.W0("lang.knp", "pipelines.knp or models.knp")


class Japanese:
    def __init__(self):
        Warnings.W0("knp.Japanese", "models.knp")


class TorchJapanese(Japanese):
    def __init__(self):
        Warnings.W1("knp.TorchJapanese")
