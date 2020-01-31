"""
A collection of handy utilities
"""

import glob
import json
import logging
import os
import tarfile
import traceback
from typing import Any, Dict, List, Tuple

import torch
from allennlp.commands.dry_run import dry_run_from_params
from allennlp.commands.predict import _PredictManager
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.params import with_fallback
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from .dataset_readers.conll18_ud_eval import UDError, evaluate, load_conllu_file
from .dataset_readers.evaluate_2019_task2 import (
    input_pairs,
    manipulate_data,
    read_conllu,
)

VOCAB_CONFIG_PATH = "config/create_vocab.json"

logger = logging.getLogger(__name__)


def merge_configs(configs: List[Params]) -> Params:
    """
    Merges a list of configurations together, with items with duplicate keys closer to the front of the list
    overriding any keys of items closer to the rear.
    :param configs: a list of AllenNLP Params
    :return: a single merged Params object
    """
    while len(configs) > 1:
        overrides, config = configs[-2:]
        configs = configs[:-2]

        if "udify_replace" in overrides:
            replacements = [
                replace.split(".") for replace in overrides.pop("udify_replace")
            ]
            for replace in replacements:
                obj = config
                try:
                    for key in replace[:-1]:
                        obj = obj[key]
                except KeyError:
                    raise ConfigurationError(f"Config does not have key {key}")
                obj.pop(replace[-1])

        configs.append(
            Params(with_fallback(preferred=overrides.params, fallback=config.params))
        )

    return configs[0]


def cache_vocab(params: Params, vocab_config_path: str = None):
    """
    Caches the vocabulary given in the Params to the filesystem. Useful for large datasets that are run repeatedly.
    :param params: the AllenNLP Params
    :param vocab_config_path: an optional config path for constructing the vocab
    """
    if "vocabulary" not in params or "directory_path" not in params["vocabulary"]:
        return

    vocab_path = params["vocabulary"]["directory_path"]

    if os.path.exists(vocab_path):
        if os.listdir(vocab_path):
            return

        # Remove empty vocabulary directory to make AllenNLP happy
        try:
            os.rmdir(vocab_path)
        except OSError:
            pass

    vocab_config_path = vocab_config_path if vocab_config_path else VOCAB_CONFIG_PATH

    params = merge_configs([params, Params.from_file(vocab_config_path)])
    params["vocabulary"].pop("directory_path", None)
    dry_run_from_params(params, os.path.split(vocab_path)[0])


def get_ud_treebank_files(
    dataset_dir: str, treebanks: List[str] = None
) -> Dict[str, Tuple[str, str, str]]:
    """
    Retrieves all treebank data paths in the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    :return: a dictionary mapping a treebank name to a list of train, dev, and test conllu files
    """
    datasets = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [
            file
            for file in sorted(os.listdir(treebank_path))
            if file.endswith(".conllu")
        ]

        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets[treebank] = (train_file, dev_file, test_file)
    return datasets


def get_ud_treebank_names(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    Retrieves all treebank names from the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :return: a list of long and short treebank names
    """
    treebanks = os.listdir(dataset_dir)
    short_names = []

    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [
            file
            for file in sorted(os.listdir(treebank_path))
            if file.endswith(".conllu")
        ]

        test_file = [file for file in conllu_files if file.endswith("test.conllu")]
        test_file = test_file[0].split("-")[0] if test_file else None

        short_names.append(test_file)

    treebanks = ["_".join(treebank.split("_")[1:]) for treebank in treebanks]

    return list(zip(treebanks, short_names))


def predict_model_with_archive(
    predictor: str,
    params: Params,
    archive: str,
    input_file: str,
    output_file: str,
    batch_size: int = 1,
):
    cuda_device = params["trainer"]["cuda_device"]

    check_for_gpu(cuda_device)
    archive = load_archive(archive, cuda_device=cuda_device)

    predictor = Predictor.from_archive(archive, predictor)

    manager = _PredictManager(
        predictor,
        input_file,
        output_file,
        batch_size,
        print_to_console=False,
        has_dataset_reader=True,
    )
    manager.run()


def predict_and_evaluate_model_with_archive(
    predictor: str,
    params: Params,
    archive: str,
    gold_file: str,
    pred_file: str,
    output_file: str,
    segment_file: str = None,
    batch_size: int = 1,
):
    if not gold_file or not os.path.isfile(gold_file):
        logger.warning(f"No file exists for {gold_file}")
        return

    segment_file = segment_file if segment_file else gold_file
    predict_model_with_archive(
        predictor, params, archive, segment_file, pred_file, batch_size
    )

    try:
        evaluation = evaluate(load_conllu_file(gold_file), load_conllu_file(pred_file))
        save_metrics(evaluation, output_file)
    except UDError:
        logger.warning(f"Failed to evaluate {pred_file}")
        traceback.print_exc()


def predict_model(
    predictor: str,
    params: Params,
    archive_dir: str,
    input_file: str,
    output_file: str,
    batch_size: int = 1,
):
    """
    Predict output annotations from the given model and input file and produce an output file.
    :param predictor: the type of predictor to use, e.g., "udify_predictor"
    :param params: the Params of the model
    :param archive_dir: the saved model archive
    :param input_file: the input file to predict
    :param output_file: the output file to save
    :param batch_size: the batch size, set this higher to speed up GPU inference
    """
    archive = os.path.join(archive_dir, "model.tar.gz")
    predict_model_with_archive(
        predictor, params, archive, input_file, output_file, batch_size
    )


def predict_and_evaluate_model(
    predictor: str,
    params: Params,
    archive_dir: str,
    gold_file: str,
    pred_file: str,
    output_file: str,
    segment_file: str = None,
    batch_size: int = 1,
):
    """
    Predict output annotations from the given model and input file and evaluate the model.
    :param predictor: the type of predictor to use, e.g., "udify_predictor"
    :param params: the Params of the model
    :param archive_dir: the saved model archive
    :param gold_file: the file with gold annotations
    :param pred_file: the input file to predict
    :param output_file: the output file to save
    :param segment_file: an optional file separate gold file that can be evaluated,
    useful if it has alternate segmentation
    :param batch_size: the batch size, set this higher to speed up GPU inference
    """
    archive = os.path.join(archive_dir, "model.tar.gz")
    predict_and_evaluate_model_with_archive(
        predictor,
        params,
        archive,
        gold_file,
        pred_file,
        output_file,
        segment_file,
        batch_size,
    )


def save_metrics(evaluation: Dict[str, Any], output_file: str):
    """
    Saves CoNLL 2018 evaluation as a JSON file.
    :param evaluation: the evaluation dict calculated by the CoNLL 2018 evaluation script
    :param output_file: the output file to save
    """
    evaluation_dict = {k: v.__dict__ for k, v in evaluation.items()}

    with open(output_file, "w") as f:
        json.dump(evaluation_dict, f, indent=4)

    logger.info("Metric     | Correct   |      Gold | Predicted | Aligned")
    logger.info("-----------+-----------+-----------+-----------+-----------")
    for metric in [
        "Tokens",
        "Sentences",
        "Words",
        "UPOS",
        "XPOS",
        "UFeats",
        "AllTags",
        "Lemmas",
        "UAS",
        "LAS",
        "CLAS",
        "MLAS",
        "BLEX",
    ]:
        logger.info(
            "{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                metric,
                100 * evaluation[metric].precision,
                100 * evaluation[metric].recall,
                100 * evaluation[metric].f1,
                "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy)
                if evaluation[metric].aligned_accuracy is not None
                else "",
            )
        )


def cleanup_training(
    serialization_dir: str, keep_archive: bool = False, keep_weights: bool = False
):
    """
    Removes files generated from training.
    :param serialization_dir: the directory to clean
    :param keep_archive: whether to keep a copy of the model archive
    :param keep_weights: whether to keep copies of the intermediate model checkpoints
    """
    if not keep_weights:
        for file in glob.glob(os.path.join(serialization_dir, "*.th")):
            os.remove(file)
    if not keep_archive:
        os.remove(os.path.join(serialization_dir, "model.tar.gz"))


def archive_bert_model(
    serialization_dir: str, config_file: str, output_file: str = None
):
    """
    Extracts BERT parameters from the given model and saves them to an archive.
    :param serialization_dir: the directory containing the saved model archive
    :param config_file: the configuration file of the model archive
    :param output_file: the output BERT archive name to save
    """
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"))

    model = archive.model
    model.eval()

    try:
        bert_model = model.text_field_embedder.token_embedder_bert.model
    except AttributeError:
        logger.warning(
            f"Could not find the BERT model inside the archive {serialization_dir}"
        )
        traceback.print_exc()
        return

    weights_file = os.path.join(serialization_dir, "pytorch_model.bin")
    torch.save(bert_model.state_dict(), weights_file)

    if not output_file:
        output_file = os.path.join(serialization_dir, "bert-finetune.tar.gz")

    with tarfile.open(output_file, "w:gz") as archive:
        archive.add(config_file, arcname="bert_config.json")
        archive.add(weights_file, arcname="pytorch_model.bin")

    os.remove(weights_file)


def evaluate_sigmorphon_model(gold_file: str, pred_file: str, output_file: str):
    """
    Evaluates the predicted file according to SIGMORPHON 2019 Task 2
    :param gold_file: the gold annotations
    :param pred_file: the predicted annotations
    :param output_file: a JSON file to save with the evaluation metrics
    """
    results_keys = ["lemma_acc", "lemma_dist", "msd_acc", "msd_f1"]

    reference = read_conllu(gold_file)
    output = read_conllu(pred_file)
    results = manipulate_data(input_pairs(reference, output))

    output_dict = {k: v for k, v in zip(results_keys, results)}

    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=4)
