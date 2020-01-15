import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import omegaconf
import srsly
from sklearn.model_selection import train_test_split
from spacy.language import Language
from spacy.scorer import Scorer

GoldParsable = Dict[str, Any]
InputData = List[Tuple[str, GoldParsable]]

log = logging.getLogger(__name__)


def create_data(cfg: omegaconf.Config) -> Tuple[InputData, InputData]:
    data = list(srsly.read_jsonl(Path(cfg.path).expanduser()))
    if cfg.ndata != -1:
        data = random.sample(data, k=cfg.ndata)
    else:
        cfg.ndata = len(data)
    train, val = train_test_split(data, test_size=cfg.val_size)
    srsly.write_jsonl(Path.cwd() / f"train-data.jsonl", train)
    srsly.write_jsonl(Path.cwd() / f"val-data.jsonl", val)
    return train, val


def report_fail(json_serializable_data: Any) -> None:
    fail_path = Path("fail.json").absolute()
    with fail_path.open("w") as f:
        json.dump(json_serializable_data, f, ensure_ascii=False)
        log.error(f"Error raised. The input data is saved in {str(fail_path)}")


def evaluate(cfg: omegaconf.Config, nlp: Language, val_data: InputData) -> Dict:
    try:
        scorer: Scorer = nlp.evaluate(val_data, batch_size=cfg.nbatch * 2)
    except Exception:
        report_fail(val_data)
        raise
    return scorer.scores
