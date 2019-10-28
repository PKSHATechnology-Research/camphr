import fire
import srsly
import spacy
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.util import minibatch
from tqdm import tqdm


def main(modeld, val_data, nval=1000, batchsize=16):
    nlp: Language = spacy.load(modeld)
    data = list(srsly.read_jsonl(val_data))
    if nval > 0:
        data = data[:nval]
    else:
        nval = len(data)
    docs = []
    golds = []
    with tqdm(total=nval) as pbar:
        for batch in minibatch(data, size=batchsize):
            texts, gold = zip(*batch)
            docs.extend(nlp.pipe(texts))
            golds.extend(gold)
            pbar.update(batchsize)

    scorer = Scorer(False)
    scorer.score(docs, golds)
    print(srsly.json_dumps(scorer))


if __name__ == "__main__":
    fire.Fire(main)

