from bedoner.utils import zero_pad


def test_zero_pad():
    a = [[1, 2], [2, 3, 4]]
    b = [[1, 2, 0], [2, 3, 4]]
    assert b == zero_pad(a)
