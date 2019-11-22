import json
from collections import Counter

import bedoner.ner_labels.labels_ene as ene
import fire

ENTITIES = "entities"


def main(jsonl):
    fail = Counter()
    with open(jsonl) as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)

            ents = d[1][ENTITIES]
            new_ents = []
            for s, e, l in ents:
                try:
                    new_ents.append((s, e, getattr(ene, l)))
                except:
                    fail[l] += 1
            d[1] = {ENTITIES: new_ents}
            # print(json.dumps(d))
    print(fail)


if __name__ == "__main__":
    fire.Fire(main)
