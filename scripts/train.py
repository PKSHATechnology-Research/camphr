#!/usr/bin/env python
import sys

# coding: utf-8

# In[1]:


from bedoner.models import *


# In[2]:


from bedoner.ner_labels.utils import make_biluo_labels
from bedoner.ner_labels.labels_irex import ALL_LABELS
from spacy.util import minibatch
from tqdm import tqdm
from pathlib import Path


# In[3]:


import json
import random


# In[13]:


data = []
with (
    Path.home() / "datasets/ner/gsk-ene-1.1-bccwj/irex/irex-positive.jsonl"
).open() as f:
    for i, line in enumerate(f):
        data.append(json.loads(line))


# In[14]:


ntrain, neval = 10000, 100
random.shuffle(data)
train_data = data[:ntrain]
val_data = data[-neval:]


# In[39]:


nlp = bert_ner(labels=make_biluo_labels(ALL_LABELS))


# # eval

# In[40]:


from spacy.gold import spans_from_biluo_tags, GoldParse
from itertools import zip_longest


def is_same(ents1, ents2):
    for e, e2 in zip_longest(ents1, ents2):
        if e != e2:
            return False
    return True


texts, golds = zip(*val_data)


def val(nlp):
    docs = list(nlp.pipe(texts))
    gs = [GoldParse(doc, **gold) for doc, gold in zip(docs, golds)]
    entsl = [spans_from_biluo_tags(doc, g.ner) for g, doc in zip(gs, docs)]
    return sum(is_same(doc.ents, ents) for doc, ents in zip(docs, entsl))


# In[41]:


# In[43]:


niter = 20
nbatch = 16
ndata = ntrain
optim = nlp.resume_training(t_total=niter, enable_scheduler=False)


# In[44]:


for i in range(niter):
    random.shuffle(train_data)
    epoch_loss = 0
    for i, batch in enumerate(minibatch(train_data, size=nbatch)):
        texts, golds = zip(*batch)
        docs = [nlp.make_doc(text) for text in texts]
        try:
            nlp.update(docs, golds, optim)
        except:
            print(texts, file=sys.stderr)
            continue
        loss = sum(doc._.loss.detach().item() for doc in docs)
        epoch_loss += loss
        print(f"{i*nbatch}/{ndata} loss: {loss}")
        if i % 10 == 9:
            acc = val(nlp)
            print(f"epoch {i} val: ", acc / neval)
    print(f"epoch {i} loss: ", epoch_loss)
    nlp.to_disk(f"irex{i}")
