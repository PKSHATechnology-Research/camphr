from camphr_core.doc import Doc


def test_type():
    def f(doc: Doc):
        for x in doc:
            a: Token = x
