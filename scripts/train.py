from dataclasses import dataclass
from typing import List, Dict
import json
from spacy.scorer import Scorer
import fire
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

@dataclass
class Config:
    data_jsonl: str = ""
    outd: str = ""
    ndata: int = 0
    niter: int = 0
    nbatch: int = 32
    label_type: str = "irex"


def get_labels(name: str) -> List[str]:
    if name == "irex":
        return irex_labels
    elif name == "ene":
        return ene_labels
    raise ValueError(f"Unknown label type: {name}")


def load_data(name: str) -> List[Dict]:
    data = []
    with open(name) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main(
    data_jsonl: str, outd: str, ndata=1000, niter=20, nbatch=32, label_type="irex"
):
    config = Config()
    config.data_jsonl = data_jsonl
    config.outd = outd
    config.ndata = ndata
    config.niter = niter
    config.nbatch = nbatch
    config.label_type = label_type

    os.mkdir(outd)
    data = random.sample(load_data(data_jsonl), k=ndata)
    train_data, val_data = train_test_split(data, test_size=0.1)

    labels = get_labels(label_type)
    with open("foo.txt", "w") as f:
        f.write("\n".join(labels))
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
                scorer: Scorer = nlp.evaluate(val_data)
                print("p: ", scorer.ents_p)
                print("r: ", scorer.ents_r)
                print("f: ", scorer.ents_f)
        print(f"epoch {i} loss: ", epoch_loss)
        nlp.to_disk(os.path.join(outd, str(i)))


if __name__ == "__main__":
    fire.Fire(main)
