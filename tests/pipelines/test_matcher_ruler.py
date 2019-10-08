import pytest
from bedoner.pipelines.matcher_ruler import MatcherRuler
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.language import Language


@pytest.fixture
def nlp():
    return English()


@pytest.fixture
def matcher(nlp):
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "hello"}, {"LOWER": "world"}]
    matcher.add("HW", None, pattern)

    pattern = [{"TEXT": "Python"}]
    matcher.add("PY", None, pattern)
    return matcher


@pytest.fixture
def phrasematcher(nlp):
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("OBAMA", None, nlp("Barack Obama"))
    return matcher


def test_matcher_ner(nlp: Language, matcher):
    ner = MatcherRuler(matcher)
    nlp.add_pipe(ner)
    text = "I am writing Hello World in Python."
    doc = nlp(text)
    assert len(doc.ents) == 2
    assert doc.ents[0].text == "Hello World"
    assert doc.ents[0].label_ == "HW"


def test_matcher_ner_with_label(nlp: Language, matcher):
    label = "WoW"
    ner = MatcherRuler(matcher, label=label)
    nlp.add_pipe(ner)
    text = "I am writing Hello World in Python."
    doc = nlp(text)
    assert len(doc.ents) == 2
    assert doc.ents[0].text == "Hello World"
    assert doc.ents[0].label_ == label
    1


def test_matcher_ner_with_matchid_to_label(nlp: Language, matcher):
    label = "FOO"
    labelmap = {nlp.vocab.strings["HW"]: label}
    ner = MatcherRuler(matcher, matchid_to_label=labelmap)
    nlp.add_pipe(ner)
    text = "I am writing Hello World in Python."
    doc = nlp(text)
    assert len(doc.ents) == 2
    assert doc.ents[0].text == "Hello World"
    assert doc.ents[0].label_ == label
    assert doc.ents[1].text == "Python"
    assert doc.ents[1].label_ == "PY"


def test_phrase_matcher_ner(nlp: Language, phrasematcher):
    ner = MatcherRuler(phrasematcher)
    nlp.add_pipe(ner)
    text = "I am writing hello world with Barack Obama."
    doc = nlp(text)
    assert len(doc.ents) == 1
    assert doc.ents[0].text == "Barack Obama"
    assert doc.ents[0].label_ == "OBAMA"
