from pathlib import Path

FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()
BERT_JA_DIR = FIXTURE_DIR / "bert-base-japanese-test"
BERT_DIR = FIXTURE_DIR / "bert-test"
XLNET_DIR = FIXTURE_DIR / "xlnet"
DATA_DIR = (Path(__file__).parent / "data/").absolute()
