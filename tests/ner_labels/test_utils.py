import pytest
from bedoner.ner_labels.utils import get_full_sekine_label


@pytest.mark.parametrize("label,full", [("BIRD", "NATURAL_OBJECT/LIVING_THING/BIRD")])
def test_get_full_sekine_label(label, full):
    assert get_full_sekine_label(label) == full
