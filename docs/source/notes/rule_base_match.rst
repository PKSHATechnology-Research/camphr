.. include:: ../replaces.txt

Rule based search
--------------------

Overview
~~~~~~~~~

Camphr provides some rule based matching pipelines:  :code:`PatternSearcher` and :code:`RegexRuler`, and :code:`MultipleRegexRuler`.
These pipelines are character-based, which means that they are more robust but could be more susceptible to false positives than token-based spaCy pipelines
`Matcher <https://spacy.io/api/matcher>`_ and `PhraseMatcher <https://spacy.io/api/phrasematcher>`_ .

Usage: RegexRuler
~~~~~~~~~~~~~~~~~

1. Create a pipe

    >>> import spacy
    >>> from camphr.pipelines import RegexRuler
    >>> nlp = spacy.blank("en")
    >>> pattern = r"[\d-]+"
    >>> pipe = RegexRuler(pattern, label="PHONE_NUMBER")
    >>> nlp.add_pipe(pipe)

2. Parse a text

    >>> text = "My phone number is 012-2345-6666"
    >>> doc = nlp(text)
    >>> print(doc.ents)
    (012-2345-6666,)
    >>> print(doc.ents[0].label_)
    PHONE_NUMBER

Usage: MultipleRegexRuler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use multiple patterns with :code:`MultipleRegexRuler`


1. Create a pipe

    >>> import spacy
    >>> from camphr.pipelines import MultipleRegexRuler
    >>> nlp = spacy.blank("en")
    >>> patterns = {"PHONE_NUMBER": r"[\d-]+", "EMAIL": "[\w.]+@[\w.]+"}
    >>> pipe = MultipleRegexRuler(patterns)
    >>> nlp.add_pipe(pipe)

2. Parse a text

    >>> text = "Phone: 012-2345-6666, email: bob@foomail.com"
    >>> doc = nlp(text)
    >>> print(doc.ents)
    (012-2345-6666, bob@foomail.com)
    >>> print([e.label_ for e in doc.ents])
    ['PHONE_NUMBER', 'EMAIL']

Usage: PatternSearcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`PatternSearcher` is useful when you want to look up words based on a large dictionary, thanks to `pyahocorasick <https://github.com/WojciechMula/pyahocorasick>`_ .
This pipeline searches words based on characters, while spaCy provides a similar pipeline `PhraseMatcher <https://spacy.io/api/phrasematcher>`_ which is a token-based searcher.

1. Create a pipe

    >>> import spacy
    >>> nlp = spacy.blank("en")
    >>> pipe = PatternSearcher.from_words(["text", "pattern searcher"]) # add words
    >>> nlp.add_pipe(pipe)

2. Parse a text

    >>> text = "This is a test text for pattern searcher."
    >>> doc = nlp(text)
    >>> doc.ents
    (text, pattern searcher)
