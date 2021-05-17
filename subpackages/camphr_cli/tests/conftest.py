import os
from pathlib import Path

import pytest


@pytest.fixture
def chdir(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)
