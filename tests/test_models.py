from contextlib import contextmanager

import pytest
from camphr.lang.torch import TorchLanguage
from camphr.models import (
    PIPELINE_ALIGNMENT,
    LangConfig,
    _align_pipeline,
    _resolve_alias,
    correct_model_config,
    create_lang,
)
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
        correct_model_config(config)
    if not e:
        assert OmegaConf.to_container(
            OmegaConf.create(modified)
        ) == OmegaConf.to_container(config)


@pytest.fixture(scope="session")
def inject_dummy():
    PIPELINE_ALIGNMENT["foo"] = ["bar", "baz"]


@pytest.mark.parametrize(
    "yml,modified",
    [
        (
            f"""
    pipeline:
        {TRANSFORMERS_NER}:
    """,
            f"""
    pipeline:
        {TRANSFORMERS_TOKENIZER}: {{}}
        {TRANSFORMERS_MODEL}: {{}}
        {TRANSFORMERS_NER}: {{}}
    """,
        ),
        (
            f"""
    pipeline:
        {TRANSFORMERS_MODEL}:
    """,
            f"""
    pipeline:
        {TRANSFORMERS_TOKENIZER}: {{}}
        {TRANSFORMERS_MODEL}: {{}}
    """,
        ),
        (
            f"""
    pipeline:
        {TRANSFORMERS_MODEL}: {{}}
        {TRANSFORMERS_TOKENIZER}: {{}}
    """,
            f"""
    pipeline:
        {TRANSFORMERS_TOKENIZER}: {{}}
        {TRANSFORMERS_MODEL}: {{}}
    """,
        ),
        (
            f"""
    pipeline:
        {TRANSFORMERS_NER}: {{}}
        {TRANSFORMERS_MODEL}: {{}}
        {TRANSFORMERS_TOKENIZER}: {{}}
    """,
            f"""
    pipeline:
        {TRANSFORMERS_TOKENIZER}: {{}}
        {TRANSFORMERS_MODEL}: {{}}
        {TRANSFORMERS_NER}: {{}}
    """,
        ),
        (
            f"""
    pipeline:
        foo: {{}}
        baz: {{}}
    """,
            f"""
    pipeline:
        bar: {{}}
        baz: {{}}
        foo: {{}}
    """,
        ),
    ],
)
def test_align_pipeline(yml, modified, inject_dummy):
    config = OmegaConf.create(yml)
    config = _align_pipeline(config)
    assert config.pipeline == OmegaConf.create(modified).pipeline


@pytest.mark.parametrize(
    "yml,modified",
    [
        (
            f"""
    pipeline:
        {TRANSFORMERS_NER}:
    pretrained: foo
    """,
            f"""
    pipeline:
        {TRANSFORMERS_TOKENIZER}: {{}}
        {TRANSFORMERS_MODEL}:
            trf_name_or_path: foo
        {TRANSFORMERS_NER}: {{}}
    pretrained: foo
    """,
        ),
        (
            f"""
    ner_label: foo
    """,
            f"""
    pipeline:
        {TRANSFORMERS_TOKENIZER}: {{}}
        {TRANSFORMERS_MODEL}:
        {TRANSFORMERS_NER}: {{}}
    ner_label: foo
    """,
        ),
    ],
)
def test_align_pipeline_and_alias(yml, modified):
    config = OmegaConf.create(yml)
    config = _resolve_alias(config)
    config = _align_pipeline(config)
    modified = OmegaConf.create(modified)
    assert list(config.pipeline) == list(modified.pipeline)
