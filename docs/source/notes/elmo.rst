.. include:: ../replaces.txt

Elmo
----

Overview
~~~~~~~~

`Elmo <https://allennlp.org/elmo>`_ is a deep contextualized word representation.
You can get contextualized **emmbedding vector** with Elmo pipeline.

Install
~~~~~~~

1. Download the model from |release-page|_

2. Install with pip:

.. parsed-literal::

    $ pip install \ |elmo-tar|\

All parameters and dependencies is installed now.

Usage
~~~~~

Elmo vector
=======================

    >>> import spacy
    >>> nlp = spacy.load("en_elmo_medium")
    >>> doc = nlp("One can deposit money at the bank")
    >>> doc.tensor
    array([[-1.8180828e+00,  4.4738990e-01, -1.3872834e-01, ...,
         4.6713990e-01,  5.8039522e-01,  7.2358000e-01]], dtype=float32)
    >>> doc[0].vector
    array([-1.8180828 ,  0.4473899 , -0.13872834, ...,  0.3815268 ,
            0.9083941 ,  0.44577932], dtype=float32)
    

Similarity of tokens
========================

Because Elmo is a contextualized vector, each token's vectors are different even if they are same word.

    >>> bank0 = nlp("One can deposit money at the bank")[-1]
    >>> bank1 = nlp("The river bank was not clean")[2]
    >>> bank2 = nlp("I withdrew cash from the bank")[-1]
    >>> print(bank0, bank1, bank2)
    bank bank bank
    >>> print(bank0.similarity(bank1), bank0.similarity(bank2))
    0.8428435921669006 0.9716585278511047

Non-English Models
==================

You can create Elmo pipe with weight and option files distributed in `AllenNLP's website <https://allennlp.org/elmo>`_

1. Download :code:`weights.hd5` and :code:`options.json` in the `website <https://allennlp.org/elmo>`_
2. Run the below:

    >>> from camphr.pipelines.elmo import Elmo
    >>> elmo = Elmo.from_elmofiles("/path/to/options.json", "/path/to/weights.hd5")
    >>> nlp = spacy.blank("en") # choose one of spaCy's languages you like
    >>> nlp.add_pipe(elmo)

API References
~~~~~~~~~~~~~~

.. autoclass:: camphr.pipelines.elmo.Elmo
    :members: from_elmofiles
