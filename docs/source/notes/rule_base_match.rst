.. include:: ../replaces.txt

Rule based search
--------------------

Overview
~~~~~~~~~

Camphr provides some rule based matching pipelines:  :code:`PatternSearcher` and :code:`RegexRuler`, and :code:`MultipleRegexRuler`.
spaCy also provides `Matcher <https://spacy.io/api/matcher>`_ and `PhraseMatcher <https://spacy.io/api/phrasematcher>`_ , which are token-based pipelines.
However the Camphr's are character-based and therefore more robust, but have more false positives.

Usage: RegexRuler
~~~~~~~~~~~~~~~~~

1. Create pipe

    >>> import spacy
    >>> from camphr.pipelines import RegexRuler
    >>> nlp = spacy.blank("en")
    >>> pattern = r"[\d-]+"
    >>> pipe = RegexRuler(pattern, label="PHONE_NUMBER")
    >>> nlp.add_pipe(pipe)

2. Parse text

    >>> text = "My phone number is 012-2345-6666"
    >>> doc = nlp(text)
    >>> print(doc.ents)
    (012-2345-6666,)
    >>> print(doc.ents[0].label_)
    PHONE_NUMBER

Usage: MultipleRegexRuler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use multiple patterns with :code:`MultipleRegexRuler`


1. Create pipe

    >>> import spacy
    >>> from camphr.pipelines import MultipleRegexRuler
    >>> nlp = spacy.blank("en")
    >>> patterns = {"PHONE_NUMBER": r"[\d-]+", "EMAIL": "[\w.]+@[\w.]+"}
    >>> pipe = MultipleRegexRuler(patterns)
    >>> nlp.add_pipe(pipe)

2. Parse text

    >>> text = "Phone: 012-2345-6666, email: bob@foomail.com"
    >>> doc = nlp(text)
    >>> print(doc.ents)
    (012-2345-6666, bob@foomail.com)
    >>> print([e.label_ for e in doc.ents])
    ['PHONE_NUMBER', 'EMAIL']

Usage: PatternSearcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`PatternSearcher` is useful when you want to look up words based on large dictionary, thanks to `pyahocorasick <https://github.com/WojciechMula/pyahocorasick>`.

spaCy provides a similar pipeline `PhraseMatcher <https://spacy.io/api/phrasematcher>`, which is a token-based searcher, but :code:`PatternSearcher` searches words based on characters.

1. Create pipe

    >>> import spacy
    >>> nlp = spacy.blank("en")
    >>> pipe = PatternSearcher.from_words(["text", "pattern searcher"]) # add words
    >>> nlp.add_pipe(pipe)

2. Parse text

    >>> text = "This is a test text for pattern searcher."
    >>> doc = nlp(text)
    >>> doc.ents
    (text, pattern searcher)
