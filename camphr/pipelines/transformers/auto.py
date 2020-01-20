from typing import NamedTuple, Type

import transformers as trf


class _TrfMap(NamedTuple):
    name: str
    config: Type[trf.PretrainedConfig]
    tokenizer: Type[trf.PreTrainedTokenizer]
    model: Type[trf.PreTrainedModel]


_TRF_MAPS = (
    _TrfMap(
        "bert-base-japanese", trf.BertConfig, trf.BertJapaneseTokenizer, trf.BertModel
    ),
    _TrfMap("xlm-roberta", trf.XLMConfig, trf.XLMTokenizer, trf.XLMModel),
    _TrfMap(
        "openai-gpt", trf.OpenAIGPTConfig, trf.OpenAIGPTTokenizer, trf.OpenAIGPTModel
    ),
    _TrfMap(
        "transfo-xl", trf.TransfoXLConfig, trf.TransfoXLTokenizer, trf.TransfoXLModel
    ),
    _TrfMap(
        "distilbert", trf.DistilBertConfig, trf.DistilBertTokenizer, trf.DistilBertModel
    ),
    _TrfMap(
        "camembert", trf.CamembertConfig, trf.CamembertTokenizer, trf.CamembertModel
    ),
    _TrfMap("albert", trf.AlbertConfig, trf.AlbertTokenizer, trf.AlbertModel),
    _TrfMap("roberta", trf.RobertaConfig, trf.RobertaTokenizer, trf.RobertaModel),
    _TrfMap("xlnet", trf.XLNetConfig, trf.XLNetTokenizer, trf.XLNetModel),
    _TrfMap("bert", trf.BertConfig, trf.BertTokenizer, trf.BertModel),
    _TrfMap("gpt2", trf.GPT2Config, trf.GPT2Tokenizer, trf.GPT2Model),
    _TrfMap("ctrl", trf.CTRLConfig, trf.CTRLTokenizer, trf.CTRLModel),
    _TrfMap("xlm", trf.XLMConfig, trf.XLMTokenizer, trf.XLMModel),
    _TrfMap("t5", trf.T5Config, trf.T5Tokenizer, trf.T5Model),
)


def _get_trf_map(text: str) -> _TrfMap:
    for item in sorted(_TRF_MAPS, key=lambda x: len(x.name), reverse=True):
        if item.name in text:
            return item
    raise ValueError(f"Couldn't find any `_TrfMap` from {text}.")


def get_trf_name(text: str) -> str:
    return _get_trf_map(text).name


def get_trf_tokenizer_cls(text: str) -> Type[trf.PreTrainedTokenizer]:
    return _get_trf_map(text).tokenizer


def get_trf_config_cls(text: str) -> Type[trf.PretrainedConfig]:
    return _get_trf_map(text).config


def get_trf_model_cls(text: str) -> Type[trf.PreTrainedModel]:
    return _get_trf_map(text).model
