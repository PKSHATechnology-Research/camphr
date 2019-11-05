#!/usr/bin/env python
"""Evaluation for the SIGMORPHON 2019 shared task, task 2.

Computes various metrics on input.

Author: Arya D. McCarthy
Last update: 2018-12-21
"""

import argparse
import logging

import numpy as np

from collections import namedtuple
from pathlib import Path

log = logging.getLogger(Path(__file__).stem)


COLUMNS = "ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC".split()
ConlluRow = namedtuple("ConlluRow", COLUMNS)
SEPARATOR = ";"


def distance(str1, str2):
    """Simple Levenshtein implementation."""
    m = np.zeros([len(str2) + 1, len(str1) + 1])
    for x in range(1, len(str2) + 1):
        m[x][0] = m[x - 1][0] + 1
    for y in range(1, len(str1) + 1):
        m[0][y] = m[0][y - 1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y - 1] == str2[x - 1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x - 1][y] + 1, m[x][y - 1] + 1, m[x - 1][y - 1] + dg)
    return int(m[len(str2)][len(str1)])


def set_equal(str1, str2):
    set1 = set(str1.split(SEPARATOR))
    set2 = set(str2.split(SEPARATOR))
    return set1 == set2


def manipulate_data(pairs):
    log.info("Lemma acc, Lemma Levenshtein, morph acc, morph F1")

    count = 0
    lemma_acc = 0
    lemma_lev = 0
    morph_acc = 0

    f1_precision_scores = 0
    f1_precision_counts = 0
    f1_recall_scores = 0
    f1_recall_counts = 0

    for r, o in pairs:
        log.debug("{}\t{}\t{}\t{}".format(r.LEMMA, o.LEMMA, r.FEATS, o.FEATS))
        count += 1
        lemma_acc += r.LEMMA == o.LEMMA
        lemma_lev += distance(r.LEMMA, o.LEMMA)
        morph_acc += set_equal(r.FEATS, o.FEATS)

        r_feats = set(r.FEATS.split(SEPARATOR)) - {"_"}
        o_feats = set(o.FEATS.split(SEPARATOR)) - {"_"}

        union_size = len(r_feats & o_feats)
        reference_size = len(r_feats)
        output_size = len(o_feats)

        f1_precision_scores += union_size
        f1_recall_scores += union_size
        f1_precision_counts += output_size
        f1_recall_counts += reference_size

    f1_precision = f1_precision_scores / (f1_precision_counts or 1)
    f1_recall = f1_recall_scores / (f1_recall_counts or 1)
    f1 = 2 * (f1_precision * f1_recall) / (f1_precision + f1_recall + 1e-20)

    return (
        100 * lemma_acc / count,
        lemma_lev / count,
        100 * morph_acc / count,
        100 * f1,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--reference", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    # Set the verbosity level for the logger. The `-v` option will set it to
    # the debug level, while the `-q` will set it to the warning level.
    # Otherwise use the info level.
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )
    return parser.parse_args()


def strip_comments(lines):
    for line in lines:
        if not line.startswith("#"):
            yield line


def read_conllu(file: Path):
    with open(file) as f:
        yield from strip_comments(f)


def input_pairs(reference, output):
    output = [o for o in output if len(o.split()) == 0 or "." not in o.split()[0]]
    reference = [r for r in reference if len(r.split()) == 0 or "." not in r.split()[0]]

    for r, o in zip(reference, output):
        assert r.count("\t") == o.count("\t"), (r.count("\t"), o.count("\t"), o)
        if r.count("\t") > 0:
            r_conllu = ConlluRow._make(r.split("\t"))
            o_conllu = ConlluRow._make(o.split("\t"))
            yield r_conllu, o_conllu


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    reference = read_conllu(args.reference)
    output = read_conllu(args.output)
    results = manipulate_data(input_pairs(reference, output))
    print(*["{0:.2f}".format(v) for v in results], sep="\t")


if __name__ == "__main__":
    main()
