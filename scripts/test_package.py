import re
import sys

import spacy
from spacy.language import Language


def main():
    fname = sys.argv[1]
    model = re.findall("/([^/]*?)-.*$", fname)[0]
    print(f"model: {model}")
    nlp = spacy.load(model)
    test_nlp(nlp)


def test_nlp(nlp: Language):
    text = "10日発表されたノーベル文学賞の受賞者をめぐり、選考機関のスウェーデン・アカデミーが批判されている。"
    print(f"test input: {text}")
    doc = nlp(text)
    for e in doc.ents:
        print(e.text, e.label_)


if __name__ == "__main__":
    main()
