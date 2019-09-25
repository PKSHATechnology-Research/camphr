import json
import torch

import os
import random
import sys
from pathlib import Path

from bedoner.models import *
from bedoner.ner_labels.labels_irex import ALL_LABELS
from bedoner.ner_labels.utils import make_biluo_labels
from spacy.util import minibatch
from tqdm import tqdm

data = []
with open(os.path.expanduser("~/datasets/ner/gsk-ene-1.1-bccwj/irex/irex.jsonl")) as f:
    for line in f:
        data.append(json.loads(line))

nlp = bert_ner(labels=make_biluo_labels(ALL_LABELS))

optim = nlp.resume_training()

niter = 5
if len(sys.argv) > 1:
    nbatch = int(sys.argv[1])
else:
    nbatch = 16
print(f"batch: {nbatch}")
ndata = 1000
data = data[:ndata]
for i in range(niter):
    random.shuffle(data)
    epoch_loss = 0
    for i, batch in enumerate(minibatch(data, size=nbatch)):
        texts, golds = zip(*batch)
        docs = [nlp.make_doc(text) for text in texts]
        nlp.update(docs, golds, optim)
        loss = sum(doc._.loss.detach().item() for doc in docs)
        print(f"{i*nbatch}/{ndata} loss: {loss}")
        epoch_loss += loss
    print(f"epoch {i} loss: ", epoch_loss)
    nlp.to_disk(f"irex{i}")
