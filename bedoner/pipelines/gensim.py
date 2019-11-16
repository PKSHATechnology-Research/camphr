import pickle
from typing import TypeVar, Union

from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models.word2vec import Word2Vec
from spacy.language import Language
from spacy.vocab import Vocab
from tqdm import tqdm

from bedoner.types import Pathlike

HasVocab = TypeVar("HasVocab", Language, Vocab)


def load_gensim_model(model_path: Pathlike) -> Union[Word2VecKeyedVectors, Word2Vec]:
    path = str(model_path)
    with open(path, "rb") as f:
        m = pickle.load(f)
    cls = type(m)
    model = cls.load(str(path))
    return model


def _set_from_word2vec_keyedvectors(
    nlp_or_vocab: HasVocab, model: Word2VecKeyedVectors
):
    if isinstance(nlp_or_vocab, Language):
        vocab = nlp_or_vocab.vocab
    elif isinstance(nlp_or_vocab, Vocab):
        vocab = nlp_or_vocab
    else:
        raise ValueError(f"Unsupported type {type(nlp_or_vocab)}")
    if isinstance(model, Word2Vec):
        model = model.wv
    for key, vec in tqdm(zip(model.index2word, model.vectors)):
        vocab.set_vector(key, vector=vec)


def set_word2vec_vectors(nlp_or_vocab: HasVocab, word2vec_model_path: Pathlike):
    model = load_gensim_model(word2vec_model_path)
    _set_from_word2vec_keyedvectors(nlp_or_vocab, model)
