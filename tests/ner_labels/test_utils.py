from pathlib import Path

import pytest

from camphr.ner_labels.utils import get_biluo_labels, get_full_sekine_label


@pytest.mark.parametrize("label,full", [("BIRD", "NATURAL_OBJECT/LIVING_THING/BIRD")])
def test_get_full_sekine_label(label, full):
    assert get_full_sekine_label(label) == full


@pytest.mark.parametrize(
    "label", [["A", "BB"], str(Path(__file__).parent / "label.json")]
)
def test_get_biluo_labels(label):
    labels = get_biluo_labels(label)
    labels2 = get_biluo_labels(labels)
    assert len(labels) == len(labels2)
    assert labels == labels2
    if isinstance(label, list):
        assert len(labels) == len(label) * 4 + 2
