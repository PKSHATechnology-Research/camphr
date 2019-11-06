from tqdm import tqdm
from bedoner.types import Pathlike
from gensim.models.keyedvectors import Word2VecKeyedVectors
from spacy.language import Language
from spacy.vocab import Vocab
from typing import TypeVar

HasVocab = TypeVar("HasVocab", Language, Vocab)


def set_word2vec_vectors(nlp_or_vocab: HasVocab, word2vec_model_path: Pathlike):
    if isinstance(nlp_or_vocab, Language):
        vocab = nlp_or_vocab.vocab
    elif isinstance(nlp_or_vocab, Vocab):
        vocab = nlp_or_vocab
    else:
        raise ValueError(f"Unsupported type {type(nlp_or_vocab)}")

    model = Word2VecKeyedVectors.load(str(word2vec_model_path))
    for key, vec in tqdm(zip(model.index2word, model.vectors)):
        vocab.set_vector(key, vector=vec)
