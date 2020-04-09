Sentencepiece as a spacy.Language
=================================

Camphr supports `Sentencepiece <https://github.com/google/sentencepiece>`_ as a :code:`spacy.Language`.
You can use Sentencepiece as you would use `en` or other languages.

Usage
-----

Pass your trained `spiece.model` file path to `spacy.blank`, as follows:

.. code:: python3

    >>> import spacy
    >>> nlp = spacy.blank("sentencepiece", meta={"tokenizer": {"model_path": "/path/to/your/spiece.model"}})

Now you can use :code:`nlp` as you normally would:

.. code:: python3

    >>> doc = nlp("I saw a  girl with a telescope.")
    >>> print(list(doc))
    [I, saw, a, girl, with, a, te, le, s, c, o, pe, .]

(The result of the tokenization depends on the :code:`spiece.model`, so you would see a different result from the above.)

The raw result of Sentencepiece can be obtained via :code:`doc._.spm_pieces_`:

.. code:: python3

    >>> print(doc._.spm_pieces_)
    ["▁I", "▁saw", "▁a", "▁girl", "▁with", "▁a", "▁", "te", "le", "s", "c", "o", "pe, "."]

You can easily get an alignment between :code:`doc._.spm_pieces_` and :code:`doc` with `pytokenizations <https://github.com/tamuhey/tokenizations/tree/master/python>`_:

.. code:: python3

    >>> import tokenizations
    >>> a2b, b2a = tokenizations.get_alignments(doc._.spm_pieces_, [token.text for token in doc])
    >>> print(a2b)
    [[0], [1], [2], [3], [4], [5], [6], [], [7], [8], [9], [10], [11], [12]]
    >>> print(doc[1:4])
    [saw, a, girl]
    >>> import itertools
    >>> print(doc._.spm_pieces_[i] for i in itertools.chain.from_iterable(b2a[1:4]))
    ["_saw", "_a", "_girl"]
