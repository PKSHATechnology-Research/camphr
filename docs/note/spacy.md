# Camphr: spaCy plugin for Transformers, Udify, KNP

Hi, I'm [Yohei Tamura](https://github.com/tamuhey), a software engineer at PKSHA Technology. I recently published a spaCy plugin called [Camphr](https://github.com/PKSHATechnology-Research/camphr/), which helps in seamless integration for a wide variety of techniques from state-of-the-art to conventional ones. You can use [Transformers](https://huggingface.co/transformers/) ,  [Udify](https://github.com/Hyperparticle/udify), [ELmo](https://allennlp.org/elmo), etc. on [spaCy](https://github.com/explosion/spaCy).

This post introduce the features and simple usage of Camphr.

## Why I chose spaCy

spaCy is an awesome NLP framework and in my opinion has following advantages:

1. [Its superior architecture](https://spacy.io/api) makes it easy to integrate different methods. (everything from deep learning to pattern searching.)
2. You can save and reload pipelines with a single command.

The first feature is very important in practice. NLP has come a long way in the last few years, but real tasks are often not as simple as end-to-end models solve them. However, if we don't use the most advanced methods, there seems to be no future. With spaCy, you can combine a variety of methods from modern to rule-based. And Camphr makes it easy, for example, to fine-tune BERT and combine regular expressions.

The second feature makes it easy to store and restore complex combined pipelines and carry them around. You can iterate faster and the service gets better and better.

## Quick Tour

Let's take a quick look at what Camphr offers.

(Table of Contents)

- Udify: BERT based Universal Dependency parser for 75 languages
- Transformers embedding vector
- Fine-tuning transformers for downstream tasks
- Elmo: deep contextualized word representation

### Udify: BERT based Universal Dependency parser for 75 languages

[Udify](https://github.com/Hyperparticle/udify) is  Universal Dependency parser for 75 languages.

First, install udify as follows.
Because the model parameters are downloaded, it may take few minutes.

```console
$ pip install https://github.com/PKSHATechnology-Research/camphr_models/releases/download/0.5/en_udify-0.5.tar.gz
```

That's all for the installation. Then use it as follows:

```python
import spacy
nlp = spacy.load("en_udify")
doc = nlp("Udify is a BERT based dependency parser")
spacy.displacy.render(doc)
```

Udify is a multilingual model, so you can parse German text *with the same model*: 

```python
doc = nlp("Deutsch kann so wie es ist analysiert werden")
spacy.displacy.render(doc) 
```

However, if you want to parse non-space-sparated languages (e.g. Chinese), you should replace tokenizer. Camphr provides helper function for this purpose:

```python
from camphr.pipelines import load_udify
nlp = load_udify("zh")
doc = nlp("也可以分析中文文本")
spacy.displacy.render(doc)
```

See the [documentation](https://camphr.readthedocs.io/en/latest/notes/udify.html) for more details.

### Transformers embedding vector

[Transformers](https://github.com/huggingface/transformers) provides state-of-the-art NLP architectures (BERT, GPT-2, ...). Camphr makes it easy to combine transformers and spaCy.
In this section, I will explain how to use Transformers models as text embedding layers. See next section for fine-tuning transformers models.

First, install camphr with pip:

```console
$ pip install camphr
```

And create `nlp` as follows:

```python
import camphr
# pass YAML or JSON or python Dict.
nlp = camphr.load(
    """
lang:
    name: en
pipeline:
    transformers_model:
        trf_name_or_path: xlnet-base-cased
"""
)
```

The configuration is parsed by [omegaconf](https://github.com/omry/omegaconf), so you can pass YAML or JSON or python Dict to `camphr.load`.
`trf_name_or_path` is a [transformers pretrained model name](https://huggingface.co/transformers/pretrained_models.html) or a directory containing your transformers pretrained model.

Transformers computes the vector representation of an input text:

```python
>>> doc = nlp("BERT converts text to vector")
>>> doc.vector
array([[-0.5427, -0.9614, -0.4943,  ...,  2.2654,  0.5592,  0.4276],
    ...

>>> doc[0].vector # token vector
array([-5.42725086e-01, -9.61372316e-01, -4.94263291e-01,  4.83379781e-01,
   -1.52603614e+00, -1.25056303e+00,  6.28554821e-01,  2.57751465e-01,
    3.44272882e-01, -3.19559097e-01, -6.80006146e-01,  1.15556490e+00,
    ... ]

>>> doc2 = nlp("Doc simlarity can be computed based on doc.tensor")
doc.similarity(doc2)
0.8234463930130005

>>> doc[0].similarity(doc2[0]) # tokens similarity
0.4105265140533447
```

Use nlp.pipe to process multiple texts at once:

```python
>>> texts = ["I am a cat.", "As yet I have no name.", "I have no idea where I was born."]
>>> docs = nlp.pipe(texts)
>>> Use nlp.to for faster processing (CUDA is required):
```
Use nlp.to for faster processing (CUDA is required):

```python
>>> import torch
>>> nlp.to(torch.device("cuda"))
>>> docs = nlp.pipe(texts)
```

See [the offcial documentation](https://camphr.readthedocs.io/en/latest/notes/transformers.html) for more information.

### Fine-tuning transformers for downstream tasks

Camphr provides a CLI to fine-tune Transformers’ pretrained models for downstream tasks, e.g. text classification and named entity recognition. The CLI is built on top of [Hydra](https://github.com/facebookresearch/hydra), which is awesome application framework.

You can fine-tune Transformers pretrained models for text classification tasks as follows:

```console
$ camphr train train.data.path="./train.jsonl" \
               model.textcat_label="./label.json" \
               model.pretrained=bert-base-cased  \
               model.lang=en
```

`train.jsonl` is a file containing training data in jsonl format:

```jsonl
["Each line contains json array", {"cats": {"POSITIVE": 0.1, "NEGATIVE": 0.9}}]
["Each array contains text and gold label", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}]
 ...
```

`label.json` contains all labels in json format:

```json
["POSITIVE", "NEGATIVE"]
```