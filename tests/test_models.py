from contextlib import contextmanager

import pytest
from camphr.lang.torch import TorchLanguage
from camphr.models import LangConfig, correct_nlp_config, create_lang
from camphr.pipelines.trf_model import TRANSFORMERS_MODEL
from camphr.pipelines.trf_ner import TRANSFORMERS_NER
from camphr.pipelines.trf_tokenizer import TRANSFORMERS_TOKENIZER
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "yml",
    [
        """
    name: en
    torch: true
    """,
        """
    name: ja_mecab
    torch: false
    """,
    ],
)
def test_create_nlp(yml):
    config: LangConfig = OmegaConf.create(yml)
    nlp = create_lang(config)
    assert not config.torch or isinstance(nlp, TorchLanguage)
    assert nlp.lang == config.name


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "yml,modified,error",
    [
        (
            f"""
    lang:
        name: en
    pipeline:
        {TRANSFORMERS_NER}:
    """,
            None,
            pytest.raises(ValueError),
        ),
        (
            f"""
    lang:
        name: en
    pipeline:
        {TRANSFORMERS_NER}:
            trf_name_or_path: foo
            labels: ['-', 'O', 'B-Foo', 'I-Foo', 'L-Foo', 'U-Foo']
    """,
            f"""
    lang:
        name: en
        torch: true
    pipeline:
        {TRANSFORMERS_TOKENIZER}:
            trf_name_or_path: foo
        {TRANSFORMERS_MODEL}:
            trf_name_or_path: foo
        {TRANSFORMERS_NER}:
            trf_name_or_path: foo
            labels: ['-', 'O', 'B-Foo', 'I-Foo', 'L-Foo', 'U-Foo']
    """,
            does_not_raise(),
        ),
        (
            f"""
    lang:
        name: en
    pipeline:
        {TRANSFORMERS_NER}:
            trf_name_or_path: foo
            labels: ["Foo"]
    """,
            f"""
    lang:
        name: en
        torch: true
    pipeline:
        {TRANSFORMERS_TOKENIZER}:
            trf_name_or_path: foo
        {TRANSFORMERS_MODEL}:
            trf_name_or_path: foo
        {TRANSFORMERS_NER}:
            trf_name_or_path: foo
            labels: ['-', 'O', 'B-Foo', 'I-Foo', 'L-Foo', 'U-Foo']
    """,
            does_not_raise(),
        ),
    ],
)
def test_correct(yml, modified, error):
    config = OmegaConf.create(yml)
    with error as e:
        correct_nlp_config(config)
    if not e:
        assert OmegaConf.to_container(
            OmegaConf.create(modified)
        ) == OmegaConf.to_container(config)
