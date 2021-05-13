import pytest

from camphr.cli.split_gold import split_gold

from ..utils import DATA_DIR, comp_jsonl


@pytest.fixture(scope="module")
def gold_jsonl():
    return str(DATA_DIR / "cli/gold.jsonl")


@pytest.fixture(scope="module")
def gold_expected_jsonl():
    return str(DATA_DIR / "cli/gold_expected.jsonl")


def test_split_gold(gold_jsonl, tmp_path, gold_expected_jsonl):
    output = str(tmp_path / "output.jsonl")
    split_gold(gold_jsonl, output, ".")
    ok, content = comp_jsonl(output, gold_expected_jsonl)
    assert ok, content
