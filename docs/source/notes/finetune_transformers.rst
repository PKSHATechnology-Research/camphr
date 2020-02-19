.. include:: ../replaces.txt

Fine tuning Transformers
------------------------------

Overview
~~~~~~~~~

Camphr provides a *command line interface* to fine-tune `Transformers <https://github.com/huggingface/transformers>`_' pretrained models for downstream tasks, e.g. *text classification* and *named entity recognition*

Text classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can fine-tune Transformers pretrained models for text classification tasks as follows:

.. code-block:: console

    $ camphr train model.task=textcat \
                   train.data.path=./train.jsonl \
                   model.labels=./label.json  \
                   model.pretrained=bert-base-cased  \
                   model.lang.name=en

Let's look at the details.

1. Prepare training data
================================================================================

Two files are required for training - :code:`train.jsonl`, :code:`label.json`.
Like `spacy <https://spacy.io/usage/training#textcat>`_, :code:`train.jsonl` contains the training data in the following  format known as `jsonl <http://jsonlines.org/>`_ :

.. code-block:: python

    ["Each line contains json array", {"cats": {"POSITIVE": 0.1, "NEGATIVE": 0.9}}]
    ["Each array contains text and gold label", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}]
     ...

:code:`label.json` is a json file defining classification labels. For example:

.. code-block:: json

    ["POSITIVE", "NEGATIVE"]



.. _choose-transformers:

2. Choose Transformers pretrained models
================================================================================

.. include:: fragments/choose_transformers_models.txt

3. Configure and Start fine-tuning
==============================================================

The following is the minimal configuration to fine-tune *bert-base-cased* with *English* tokenizer.

.. code-block:: console

    $ camphr train model.task=textcat \
                   train.data.path="./train.jsonl" \
                   model.labels="./label.json" \
                   model.pretrained=bert-base-cased  \
                   model.lang.name=en

Of course, you can also use non-English languages, by changing *model.lang.name*:

.. code-block:: console

    $ camphr train model.task=textcat \
                   train.data.path="./train.jsonl" \
                   model.labels="./label.json" \
                   model.pretrained=bert-base-multilingual-cased  \
                   model.lang.name=ja # Japanese

.. |use-cuda| replace:: If CUDA is available, it will be enabled automatically.

:note: |use-cuda|

.. include:: fragments/note-output-dir.txt

.. include:: fragments/to-advance-config.txt



4. Use fine-tuned models
================================

.. include:: fragments/use-model.txt

Multilabel Text classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Camphr enables you to fine-tune transformers pretrained model for multi-labels textcat tasks:

.. code-block:: console

    $ camphr train task=multilabel_textcat \
                   train.data.path=./train.jsonl \
                   model.labels=./label.json  \
                   model.pretrained=bert-base-cased  \
                   model.lang.name=en

Let's look at the details.

1. Prepare training data
================================================================================

Two files are required for training - :code:`train.jsonl`, :code:`label.json`.
Like `spacy <https://spacy.io/usage/training#textcat>`_, :code:`train.jsonl` contains the training data in the following  format known as `jsonl <http://jsonlines.org/>`_ :

.. code-block:: python

    ["Each line contains json array", {"cats": {"A": 0.1, "B": 0.8, "C": 0.8}}]
    ["Each array contains text and gold label", {"cats": {"A": 0.1, "B": 0.9, "C": 0.8}}]
     ...
    
Because the task is multi-labels, the total score on each labels doesn't have to be 1.

:code:`label.json` is a json file defining classification labels. For example:

.. code-block:: json

    ["A", "B", "C"]


.. _choose-transformers:

2. Choose Transformers pretrained models
================================================================================

.. include:: fragments/choose_transformers_models.txt

3. Configure and Start fine-tuning
==============================================================

The following is the minimal configuration to fine-tune *bert-base-cased* with *English* tokenizer.

.. code-block:: console

    $ camphr train task=multilabel_textcat \
                   train.data.path="./train.jsonl" \
                   model.labels="./label.json" \
                   model.pretrained=bert-base-cased  \
                   model.lang.name=en

Of course, you can also use non-English languages, by changing *model.lang.name*:

.. code-block:: console

    $ camphr train task=multilabel_textcat \
                   train.data.path="./train.jsonl" \
                   model.labels="./label.json" \
                   model.pretrained=bert-base-multilingual-cased  \
                   model.lang.name=ja # Japanese

.. |use-cuda| replace:: If CUDA is available, it will be enabled automatically.

:note: |use-cuda|

.. include:: fragments/note-output-dir.txt

.. include:: fragments/to-advance-config.txt



4. Use fine-tuned models
================================

.. include:: fragments/use-model.txt

Named entity recognition
~~~~~~~~~~~~~~~~~~~~~~~~

You can also fine-tune Transformers models for `named entity recognition <https://en.wikipedia.org/wiki/Named-entity_recognition>`_ with Camphr's CLI:

.. code-block:: console

    $ camphr train model.task=ner \
                   train.data.path="./train.jsonl" \
                   model.labels="./label.json" \
                   model.pretrained=bert-base-cased  \
                   model.lang.name=en

Let's look at the details.

1. Prepare training data
================================================================================

Two files are required for training - :code:`train.jsonl`, :code:`label.json`.
Like `spacy <https://spacy.io/usage/training#textcat>`_, :code:`train.jsonl` contains the training data in the following  format known as `jsonl <http://jsonlines.org/>`_ :

.. code-block:: python

    ["I live in Japan.", {"entities": [[10, 15, "LOCATION"]] }]
    ["Today is January 30th", {"entities": [[9, 21, "DATE"]] }]
     ...

*"entities"* is an array containing arrays that consist of :code:`[start_char_pos, end_char_pos, label_type]`.

:code:`label.json` is a json file defining classification labels. For example:

.. code-block:: json

    ["DATE", "PERSON", "ORGANIZATION"]

2. Choose Transformers pretrained model
================================================================================

.. include:: fragments/choose_transformers_models.txt



3. Configure and Start fine-tune
==============================================================

The following is the minimal configuration to fine-tune *bert-base-cased* with *English* tokenizer.

.. code-block:: console

    $ camphr train model.task=ner \
                   train.data.path="./train.jsonl" \
                   model.labels="./label.json" \
                   model.pretrained=bert-base-cased  \
                   model.lang.name=en

You can also use *non-English languages*, by changing *model.lang.name*:

.. code-block:: console

    $ camphr train model.task=ner \
                   train.data.path="./train.jsonl" \
                   model.label="./label.json" \
                   model.pretrained=bert-base-multilingual-cased  \
                   model.lang.name=ja # Japanese

:note: |use-cuda|

.. include:: fragments/to-advance-config.txt

.. include:: fragments/note-output-dir.txt

4. Use fine-tuned models
================================

.. include:: fragments/use-model.txt

.. _advance-config:

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

Camphr uses `Hydra <https://hydra.cc/>`_ as a training configuration system, and the configuration can be customized in Hydra's convention.

First, let's see a sample configuration:

.. code-block:: console

    $ camphr train example=ner --cfg job

    model:
        lang:
            name: en
        ner_label: ~/irex.json
        pipeline: null
        pretrained: bert-base-cased
    train:
        data:
            ndata: -1
            path: ~/train.jsonl
            val_size: 0.1
        nbatch: 16
        niter: 10
        optimizer:
            class: transformers.optimization.AdamW
            params:
                eps: 1.0e-08
                lr: 2.0e-05
        scheduler:
            class: transformers.optimization.get_linear_schedule_with_warmup
            params:
                num_training_steps: 7
                num_warmup_steps: 3

As you can see, the configuration is defined in `YAML <https://en.wikipedia.org/wiki/YAML>`_ format.

You can override values in the loaded configuration from the commend line.
For example, in order to replace :code:`model.lang.name` with :code:`ja`, pass :code:`model.lang.name=ja` in CLI:

.. code-block:: console

    $ camphr train model.lang.name=ja

    model:
        lang:
            name: ja
    ...

Pass yaml
=========

The more items you wish to override, the more tedious it becomes for you to enter them on the command line.

You can rewrite the configuration with yaml file instead of command line options.
For example, prepare :code:`user.yaml` as follows:

.. code-block:: yaml

    model:
        lang:
            name: ja
        optimizer:
            class: transformers.optimization.AdamW
            params:
                eps: 1.0e-05
                lr: 1.0e-03
    train:
        data:
            ndata: -1
            path: ~/train.jsonl
            val_size: 0.1
        nbatch: 128
        niter: 30

And pass the yaml to CLI as follows:

.. code-block:: console

    $ camphr train user=user.yaml

.. seealso::

    :doc:`transformers`: For use embedding vector, without fine-tuning
