import shutil


def check_juman():
    return shutil.which("juman") is not None


def check_knp():
    return shutil.which("knp") is not None


def check_mecab():
    return shutil.which("mecab") is not None
