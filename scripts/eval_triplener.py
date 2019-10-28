# adhoc script
import re
from scripts.train_triplener import get_top_label, get_second_top_label
import fire
import srsly
import spacy
import torch
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.util import minibatch
from tqdm import tqdm
from bedoner.pipelines.trf_ner import TrfForNamedEntityRecognitionBase
from spacy.gold import GoldParse


CONVERT_LABEL = "convert_label"


def create_nlp(modeld):
    nlp: Language = spacy.load(modeld)
    for k, v in nlp.pipeline:
        if k.endswith("ner"):
            v.add_user_hook(CONVERT_LABEL, get_top_label)
        elif k.endswith("ner2"):
            v.add_user_hook(
                CONVERT_LABEL, lambda x: get_second_top_label(x, add_pref=False)
            )
    return nlp


def get_ner_names(nlp):
    names = []
    for k, v in nlp.pipeline:
        if re.search("ner\d*$", k):
            names.append(k)
    return names


def modify_gold(data, fn):
    key = "entities"
    new_data = []
    for doc, gold in data:
        ents = []
        for s, e, l in gold["entities"]:
            ents.append((s, e, fn(l)))
        new_data.append((doc, {key: ents}))
    return new_data


def eval(nlp: Language, data, batchsize, fn):
    nval = len(data)
    scorer = Scorer(pipeline=nlp.pipeline)
    with tqdm(total=nval) as pbar:
        for batch in minibatch(data, size=batchsize):
            scorer = nlp.evaluate(modify_gold(batch, fn), scorer=scorer)
            pbar.update(batchsize)
    print(srsly.json_dumps(scorer))


def main(modeld, val_data, nval=-1, batchsize=16, gpu=False):
    nlp: Language = create_nlp(modeld)
    if gpu and torch.cuda.is_available():
        nlp.to(torch.device("cuda"))
    data = list(srsly.read_jsonl(val_data))
    if nval > 0:
        data = data[:nval]
    else:
        nval = len(data)
    names = get_ner_names(nlp)
    for name in names:
        print(name)
        with nlp.disable_pipes(*list(set(names) - {name})):
            fn = nlp.get_pipe(name).user_hooks.get(CONVERT_LABEL, (lambda x: x))
            eval(nlp, data, batchsize, fn)
        print()


if __name__ == "__main__":
    fire.Fire(main)

