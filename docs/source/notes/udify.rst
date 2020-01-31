.. include:: ../replaces.txt

Udify
------

Overview
~~~~~~~~~

`Udify <https://arxiv.org/abs/1904.02099>`_ is a *BERT based, multilingual multi-task model* capable of predicting **universal part-of-speech**, **morphological features**, **lemmas**, and **dependency trees** simultaneously for **75 languages**.

Install
~~~~~~~

1. Download the model from |release-page|_

2. Install with pip:

.. parsed-literal::

    $ pip install \ |udify-tar|\

All parameters and dependencies is installed now.

Usage
~~~~~

    >>> import spacy
    >>> nlp = spacy.load("udify")
    >>> doc = nlp("Udify is a BERT based dependency parser")
    >>> spacy.displacy.render(doc)

    .. image:: udify_dep_en.png

You can use :code:`nlp` for space delimited languages:

    >>> nlp("Deutsch kann so wie es ist analysiert werden")

    .. image:: udify_dep_de.png

Use udify with non-space-delimited languages
============================================

You can use Udify by replacing the tokenizer.
capmhr provides some useful functions for this purpose: :code:`load_udify`:

    >>> from capmhr.pipelines import load_udify
    >>> nlp = load_udify("ja", punct_chars=["。"])
    >>> nlp("日本語も解析可能です")

    .. image:: udify_dep_ja.png

.. note:: To use Udify with Japanese, |require-mecab|

API References
~~~~~~~~~~~~~~

.. autofunction:: camphr.pipelines.udify.load_udify
.. autofunction:: camphr.pipelines.udify.load_udify_pipes
.. autoclass:: camphr.pipelines.udify.Udify
