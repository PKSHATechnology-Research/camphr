def check_mecab() -> bool:
    try:
        import MeCab  # noqa
    except ImportError:
        return False
    return True
