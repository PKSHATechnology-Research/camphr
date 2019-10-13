from pathlib import Path
import pytest
from bedoner.cli import split_gold
from .utils import comp_jsonl
import filecmp


@pytest.fixture(scope="module")
def gold_jsonl(DATADIR: Path):
    return str(DATADIR / "cli/gold.jsonl")


@pytest.fixture(scope="module")
def gold_expected_jsonl(DATADIR: Path):
    return str(DATADIR / "cli/gold_expected.jsonl")


def test_split_gold(gold_jsonl, tmp_path, gold_expected_jsonl):
    output = str(tmp_path / "output.jsonl")
    split_gold(gold_jsonl, output, ".")
    ok, content = comp_jsonl(output, gold_expected_jsonl)
    assert ok, content
