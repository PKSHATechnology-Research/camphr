def check_juman() -> bool:
    try:
        import pyknp  # noqa
    except ImportError:
        return False
    return True


def check_knp() -> bool:
    return check_juman()


def check_mecab() -> bool:
    try:
        import MeCab  # noqa
    except ImportError:
        return False
    return True
