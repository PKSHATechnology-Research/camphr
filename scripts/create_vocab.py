"""Create spacy vocab from text file containing words line by line

Text file style: one word per line.
```
foo
bar
baz
...
```
"""
import fire
from spacy.strings import StringStore
from spacy.vocab import Vocab


def main(text_path: str, output_dir_path: str):
    with open(text_path) as f:
        vs = []
        for line in f:
            vs.append(line[:-1])
    print(f"{text_path} loaded")

    s = StringStore(vs)
    v = Vocab(strings=s)
    v.to_disk(output_dir_path)
    print(f"wrote into {output_dir_path}!")


if __name__ == "__main__":
    fire.Fire(main)
