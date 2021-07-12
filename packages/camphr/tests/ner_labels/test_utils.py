from pathlib import Path

import pytest

from camphr.ner_labels.utils import get_ner_labels


@pytest.mark.parametrize(
    "label", [["A", "BB"], str(Path(__file__).parent / "label.json")]
)
def test_get_ner_labels(label):
    labels = get_ner_labels(label)
    labels2 = get_ner_labels(labels)
    assert len(labels) == len(labels2)
    assert labels == labels2
    if isinstance(label, list):
        assert len(labels) == len(label) * 2 + 2
