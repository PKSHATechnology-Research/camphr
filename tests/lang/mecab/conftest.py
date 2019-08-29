import pytest
from bedoner.lang.mecab import Japanese


@pytest.fixture(scope="module")
def mecab_tokenizer():
    return Japanese.Defaults.create_tokenizer(opt="-d /usr/local/lib/mecab/dic/ipadic")
