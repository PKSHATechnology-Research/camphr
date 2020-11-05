.. include:: ../replaces.txt

Udify
------

Overview
~~~~~~~~~

`Udify <https://arxiv.org/abs/1904.02099>`_ is a *BERT based, multilingual multi-task model* capable of predicting **universal part-of-speech**, **morphological features**, **lemmas**, and **dependency trees** simultaneously for **75 languages**.

Installation
~~~~~~~~~~~~

1. Download the model from |model-page|_

2. Install with pip:

.. parsed-literal::

    $ pip install \ |udify-tar|\

All parameters and dependencies are installed now.

Usage
~~~~~

    >>> import spacy
    >>> nlp = spacy.load("en_udify")
    >>> doc = nlp("Udify is a BERT based dependency parser")
    >>> spacy.displacy.render(doc)

    .. image:: udify_dep_en.png

Now you can use :code:`nlp` for space-delimited languages such as English and German:

    >>> doc = nlp("Deutsch kann so wie es ist analysiert werden")
    >>> spacy.displacy.render(doc)

    .. image:: udify_dep_de.png

Use udify with non-space-delimited languages
============================================

Switching the tokenizer allows you to use Udify for non-space-delimited languages such as Japanese.
Camphr offers a useful function for this purpose: :code:`load_udify`:

    >>> from camphr_allennlp.udify import load_udify
    >>> nlp = load_udify("ja", punct_chars=["。"])
    >>> doc = nlp("日本語も解析可能です")
    >>> spacy.displacy.render(doc)

    .. image:: udify_dep_ja.png

.. note:: To use Udify with Japanese, |require-mecab|
