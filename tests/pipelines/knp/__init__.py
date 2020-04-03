import pytest

from ...utils import check_knp

pytestmark = pytest.mark.skipif(not check_knp(), reason="knp is not always necessary")
