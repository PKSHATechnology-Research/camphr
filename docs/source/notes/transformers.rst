.. include:: ../replaces.txt

Transformers
----------------------------

Overview
~~~~~~~~~

Camphr provides `Transformers <https://github.com/huggingface/transformers>`_ as a spaCy pipeline.
You can use the transformers output with spaCy interface, or :doc:`finetune it for downstream tasks <finetune_transformers>`.

In this section, we explain how to use Transformers model as *text embedding layer*.
See :doc:`finetune_transformers` for fine-tuning transformers models.


Install
~~~~~~~

.. parsed-literal::

    \ |install-camphr|\


Usage
~~~~~

Create and add :code:`transformers_tokenizer` and :code:`transformers_model` to :code:`nlp`

    >>> nlp = spacy.blank("en")
    >>> config = {"trf_name_or_path": "bert-base-cased"}
    >>> nlp.add_pipe(nlp.create_pipe("transformers_tokenizer", config=config))
    >>> nlp.add_pipe(nlp.create_pipe("transformers_model", config=config))

Or, you can get this :code:`nlp` more easily with :doc:`camphr.load <camphr_load>`

    >>> import camphr
    >>> nlp = camphr.load(
    >>> """
    >>> lang:
    >>>     name: en
    >>> pipeline:
    >>>     transformers_model:
    >>>         trf_name_or_path: xlnet-base-cased # Other than BERT can be used.
    >>> """
    >>> ) # pass config that omegaconf can parse (YAML, Json, Dict...)

Transformers computes the vector representation of input text:

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

Use :code:`nlp.pipe` to process multiple texts at once:

    >>> texts = ["I am a cat.", "As yet I have no name.", "I have no idea where I was born."]
    >>> docs = nlp.pipe(texts)

Use for faster processing (CUDA is required):

    >>> import torch
    >>> nlp.to(torch.device("cuda"))
    >>> docs = nlp.pipe(texts)


Load local model
================

You can use models stored in local directory:

    >>> nlp = load(
    >>> """
    >>> lang:
    >>>     name: en
    >>> pipeline:
    >>>     transformers_model:
    >>>         trf_name_or_path: /path/to/your/model/directory
    >>> """
    >>> )

.. seealso::

    :doc:`finetune_transformers`: For downstream tasks, such as named entity recognition or text classification.
