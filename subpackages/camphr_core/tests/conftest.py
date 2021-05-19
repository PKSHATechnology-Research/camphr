from pathlib import Path

import pytest
import torch

FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()


@pytest.fixture(scope="session")
def spiece_path():
    return str(FIXTURE_DIR / "spiece.model")


@pytest.fixture(scope="session")
def spiece(spiece_path):
    import sentencepiece as spm

    s = spm.SentencePieceProcessor()
    s.load(spiece_path)
    return s


@pytest.fixture(scope="session", params=["cuda", "cpu"])
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("cuda is required")
    return torch.device("cuda")
