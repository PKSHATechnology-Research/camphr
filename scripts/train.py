from typing import List, Dict
import json
import random
import sys
from itertools import zip_longest
from pathlib import Path

from spacy.gold import GoldParse, spans_from_biluo_tags
from spacy.util import minibatch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from bedoner.models import *
from bedoner.ner_labels.labels_irex import ALL_LABELS as irex_labels
from bedoner.ner_labels.labels_ene import ALL_LABELS as ene_labels
from bedoner.ner_labels.utils import make_biluo_labels, make_bio_labels

def get_labels(name: str) -> List[str]:
    if "irex":
        return irex_labels
    elif "ene":
        return ene_labels
    raise ValueError(f"Unknown label type: {name}")

def load_data(name: str) -> List[Dict]:
    data = []
    with open(name) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main(data_jsonl:str,outd:str, ndata=1000,niter=20,nbatch=32,label_type="irex"):
    os.mkdir(outd)
    data=random.sample(load_data(data_jsonl), k=ndata)
    train_data, val_data=train_test_split(data,test_size=0.1)

    labels=get_labels(label_type)
    nlp = bert_ner(labels=make_biluo_labels(labels))

    optim = nlp.resume_training(t_total=niter, enable_scheduler=False)

    for i in range(niter):
        random.shuffle(train_data)
        epoch_loss = 0
        for j, batch in enumerate(minibatch(train_data, size=nbatch)):
            texts, golds = zip(*batch)
            docs = [nlp.make_doc(text) for text in texts]
            nlp.update(docs, golds, optim)
            loss = sum(doc._.loss.detach().item() for doc in docs)
            epoch_loss += loss
            print(f"{j*nbatch}/{ndata} loss: {loss}")
            if j % 10 == 9:
                score = nlp.evaluate(val_data, verbose=True)
        print(f"epoch {i} loss: ", epoch_loss)
        nlp.to_disk(os.path.join(outd, str(i)))
