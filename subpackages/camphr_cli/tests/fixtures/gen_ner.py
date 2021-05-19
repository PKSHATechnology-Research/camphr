import random

from camphr_core.utils import dump_jsonl

ALL_LABELS = [
    "ARTIFACT",
    "DATE",
    "LOCATION",
    "MONEY",
    "ORGANIZATION",
    "PERCENT",
    "PERSON",
    "TIME",
]


def gen_ner_span(length: int):
    result = []
    cur = 0
    while cur < length:
        i, cur = cur, random.randint(cur, length)
        if random.choice([True, False]):
            result.append([i, cur, random.choice(ALL_LABELS)])
    return result


def main():
    text = input()
    MAX_LENGTH = 100
    lines = []
    while text:
        length = random.randint(1, min(MAX_LENGTH, len(text)))
        cur, text = text[:length], text[length:]
        ners = gen_ner_span(length)
        lines.append([cur, {"entities": ners}])

    with open("-") as f:
        dump_jsonl(f, lines)
