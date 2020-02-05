import pytest

import camphr

pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def nlp():
    return camphr.load(
        """
    lang:
        name: en
    pipeline:
        transformers_model:
            trf_name_or_path: xlm-roberta-base
    """
    )


@pytest.mark.xfail(
    reason="Bug of Transformers: https://github.com/huggingface/transformers/issues/2727"
)
def test_pipe(nlp):
    texts = [
        "I am a cat.",
        "As yet I have no name.",
        "I have no idea where I was born.",
    ]
    docs = nlp.pipe(texts)
    list(docs)
