import spacy
import fire
import tempfile
from bedoner.lang.torch_mixin import TorchLanguageMixin


def main(path: str, dst: str):
    nlp = spacy.load(path)
    nlp.lang = "torch_" + nlp.lang
    with tempfile.TemporaryDirectory() as d:
        nlp.to_disk(d)
        nlp = spacy.load(d)
        assert isinstance(nlp, TorchLanguageMixin)
    nlp.to_disk(dst)


if __name__ == "__main__":
    fire.Fire(main)
