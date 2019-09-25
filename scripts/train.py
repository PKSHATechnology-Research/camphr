#%%
import json

# To add a new cell, type '#%%'
#%%
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
import random
import sys

from bedoner.models import *
from bedoner.ner_labels.labels_irex import ALL_LABELS
from bedoner.ner_labels.utils import make_biluo_labels
from spacy.util import minibatch
from tqdm import tqdm

try:
    os.chdir(os.path.join(os.getcwd(), "scripts"))
    print(os.getcwd())
except:
    pass


#%%
data = []
with open("../data/gsk-ene-1.1-bccwj/irex/irex.jsonl") as f:
    for line in f:
        data.append(json.loads(line))


#%%
nlp = bert_ner(labels=make_biluo_labels(ALL_LABELS))


#%%
optim = nlp.resume_training()


#%%
niter = 5
if len(sys.argv) > 1:
    nbatch = int(sys.argv[1])
else:
    nbatch = 16
print(f"batch: {nbatch}")
ndata = 10000
data = data[:ndata]
for i in range(niter):
    random.shuffle(data)
    for batch in tqdm(minibatch(data, size=nbatch)):
        texts, golds = zip(*batch)
        try:
            nlp.update(texts, golds, optim, debug=True)
        except:
            print(texts)
    nlp.to_disk(f"irex{i}")
