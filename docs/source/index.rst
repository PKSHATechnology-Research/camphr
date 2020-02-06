.. camphr documentation master file, created by
   sphinx-quickstart on Wed Jan 29 22:55:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: replaces.txt

Camphr
==================================

Camphr is a spaCy plugin to easily use `Transformers <https://huggingface.co/transformers/>`_ ,  `Udify <https://github.com/Hyperparticle/udify>`_, `ELmo <https://allennlp.org/elmo>`_, etc.

Features
~~~~~~~~

* `Transformers <https://huggingface.co/transformers/>`_ with `spaCy <https://spacy.io/>`_ - :doc:`Fine tuning <notes/finetune_transformers>`, :doc:`Embedding vector <notes/transformers>`



* `Udify <https://github.com/Hyperparticle/udify>`_ - BERT based multitask model in 75 languages

* `Elmo <https://allennlp.org/elmo>`_ - Deep contextualized word representations

* Rule base matching with `Aho-Corasick <https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm>`_, Regex

* (for Japanese) `KNP <http://nlp.ist.i.kyoto-u.ac.jp/index.php?KNP>`_

Installation
~~~~~~~~~~~~

Just pip install:

.. parsed-literal::

    |install-camphr|

Camphr requires Python3.6 or newer.

Quick tour
~~~~~~~~~~

:doc:`Transformers for text embedding <notes/transformers>`
-----------------------------------------------------------------------------

    >>> doc = nlp("BERT converts text to vector")
    >>> doc.tensor
    tensor([[-0.5427, -0.9614, -0.4943,  ...,  2.2654,  0.5592,  0.4276],
        ...
        [ 0.2395,  0.5651, -0.0630,  ..., -0.5684,  0.3808,  0.2490]])
    <BLANKLINE>
    >>> doc[0].vector # token vector
    array([-5.42725086e-01, -9.61372316e-01, -4.94263291e-01,  4.83379781e-01,
       -1.52603614e+00, -1.25056303e+00,  6.28554821e-01,  2.57751465e-01,
        3.44272882e-01, -3.19559097e-01, -6.80006146e-01,  1.15556490e+00,
        ... ]
    <BLANKLINE>
    >>> doc2 = nlp("Doc simlarity can be computed based on doc.tensor")
    >>> doc.similarity(doc2)
    0.8234463930130005
    <BLANKLINE>
    >>> doc[0].similarity(doc2[0]) # tokens similarity
    0.4105265140533447

:doc:`Fine-tune Transformers for NER and text classification <notes/finetune_transformers>`
-------------------------------------------------------------------------------------------

.. code-block:: console

    $ camphr train train.data.path="./train.jsonl" \
                   textcat_label="./label.json" \
                   pretrained=bert-base-cased  \
                   lang=en


    >>> import spacy
    >>> nlp = spacy("./output/2020-01-30/19-31-23/models/0")
    >>> doc = nlp("Fine-tune Transformers and use it as a spaCy pipeline")
    >>> print(doc.ents)
    [Transformers, spaCy]


:doc:`Udify - BERT based dependency parser for 75 languages <notes/udify>`
--------------------------------------------------------------------------

    >>> nlp = spacy.load("udify")
    >>> doc = nlp("Udify is a BERT based dependency parser")
    >>> spacy.displacy.render(doc)

    .. image:: notes/udify_dep_en.png

    >>> nlp("Deutsch kann so wie es ist analysiert werden")

    .. image:: notes/udify_dep_de.png



:doc:`Elmo - Deep contextualized word representations <notes/elmo>`
-------------------------------------------------------------------

    >>> nlp = spacy.load("en_elmo_medium")
    >>> doc = nlp("One can deposit money at the bank")
    >>> doc.tensor
    array([[-1.8180828e+00,  4.4738990e-01, -1.3872834e-01, ...,
         4.6713990e-01,  5.8039522e-01,  7.2358000e-01]], dtype=float32)
    >>> doc[0].vector
    array([-1.8180828 ,  0.4473899 , -0.13872834, ...,  0.3815268 ,
            0.9083941 ,  0.44577932], dtype=float32)


See the tutorials below for more details.

Tutorials
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   notes/transformers
   notes/finetune_transformers
   notes/udify
   notes/elmo
   notes/rule_base_match
   notes/camphr_load
   notes/knp

