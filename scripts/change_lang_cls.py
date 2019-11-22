import tempfile
from pathlib import Path
from typing import Optional, Union

import fire
import spacy

Pathlike = Union[str, Path]


TEXT = "今日はいい天気だ"


def main(modeldir: Pathlike, lang: str, savedir: Optional[Pathlike] = None):
    """Change language class of spacy's model saved in modeldir.

    Args:
        modeldir: directory containing spacy's model
        lang: Language identifire you want to change to.
        savedir: directory to save nlp model.
    """
    print(f"load from: {modeldir}")
    nlp = spacy.load(modeldir)
    print(f"change lang from {nlp.meta['lang']} to {lang}")
    nlp.meta["lang"] = lang
    if not savedir:
        savedir = tempfile.TemporaryDirectory().name
    nlp.to_disk(savedir)
    print(f"saved: {savedir}")
    print("test restore")
    nlp = spacy.load(savedir)
    assert nlp.meta["lang"] == lang
    nlp(TEXT)
    print("test passed")


if __name__ == "__main__":
    fire.Fire(main)
