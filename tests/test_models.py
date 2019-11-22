from bedoner.models import trf_model, trf_ner


def test_freeze_model(pretrained):
    nlp = trf_model("mecab", pretrained, freeze=True)
    pipe = nlp.pipeline[-1][1]
    assert pipe.cfg["freeze"]


def test_freeze_ner(pretrained):
    nlp = trf_ner("mecab", pretrained, freeze=True, labels=["foo"])
    pipe = nlp.pipeline[-2][1]
    assert pipe.cfg["freeze"]
