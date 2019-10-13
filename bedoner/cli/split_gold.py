from typing import Any, Dict, IO, Tuple
import fire
import json
from bedoner.utils import split_keepsep
import tqdm


class K:
    ents = "entities"


def _split_gold_jsonl(reader: IO[str], writer: IO[str], sep: str, verbose=False):
    for line in tqdm.tqdm(reader, disable=not verbose):
        data: Tuple[str, Dict[str, Any]] = json.loads(line)
        assert len(data) == 2, line
        text, gold = data
        assert isinstance(text, str), text
        assert isinstance(gold, dict), gold

        texts = split_keepsep(text, sep)
        if len(texts) == 1:
            writer.write(line + "\n")
            continue

        gold_ents = gold[K.ents]
        offset = 0
        for t in texts:
            next_offset = offset + len(t)
            ents = [
                (start - offset, end - offset, label)
                for start, end, label in gold_ents
                if start >= offset and end <= next_offset
            ]
            new_line = [t, {K.ents: ents}]
            writer.write(json.dumps(new_line) + "\n")
            offset = next_offset


def split_gold(fname: str, output: str, sep: str):
    with open(fname) as f, open(output, "w") as fw:
        _split_gold_jsonl(f, fw, sep)


if __name__ == "__main__":
    fire.Fire(split_gold)
