from pathlib import Path


FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()
BERT_DIR = FIXTURE_DIR / "bert-test"
BERT_JA_DIR = FIXTURE_DIR / "bert-base-japanese-test"
XLNET_DIR = FIXTURE_DIR / "xlnet"
DATA_DIR = (Path(__file__).parent / "data/").absolute()

TRF_TESTMODEL_PATH = [str(BERT_JA_DIR), str(XLNET_DIR), str(BERT_DIR)]
LARGE_MODELS = {"albert-base-v2"}
