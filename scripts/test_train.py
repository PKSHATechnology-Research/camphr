from train import Config
from pathlib import Path


def test_main():
    cfg = Config()
    cfg.data = Path(__file__) / "../tests/data/ner/gold-ene.jsonl"
