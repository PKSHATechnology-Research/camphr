"""Package ner_labels defines labels of named entity recognition and some utilities."""
from .labels_ene import ALL_LABELS as ene_labels
from .labels_irex import ALL_LABELS as irex_labels

LABELS = {"ene": ene_labels, "irex": irex_labels}

__all__ = ["LABELS"]
