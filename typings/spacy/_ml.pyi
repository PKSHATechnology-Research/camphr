"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

from thinc import describe
from thinc.api import layerize
from thinc.describe import Biases, Dimension, Gradient, Synapses
from thinc.neural._classes.affine import _set_dimensions_if_needed
from thinc.v2v import Affine, Model

VECTORS_KEY = "spacy_pretrained_vectors"
USE_MODEL_REGISTRY_TOK2VEC = False

def cosine(vec1, vec2): ...
def create_default_optimizer(ops, **cfg): ...
@layerize
def _flatten_add_lengths(seqs, pad=..., drop=...): ...
def _zero_init(model): ...
def with_cpu(ops, model):
    """Wrap a model that should run on CPU, transferring inputs and outputs
    as necessary."""
    ...

def _to_cpu(X): ...
def _to_device(ops, X): ...

class extract_ngrams(Model):
    def __init__(self, ngram_size, attr=...):
        self.ngram_size = ...
        self.attr = ...
    def begin_update(self, docs, drop=...): ...

@describe.on_data(
    _set_dimensions_if_needed, lambda model, X, y: model.init_weights(model)
)
@describe.attributes(
    nI=Dimension("Input size"),
    nF=Dimension("Number of features"),
    nO=Dimension("Output size"),
    nP=Dimension("Maxout pieces"),
    W=Synapses("Weights matrix", lambda obj: (obj.nF, obj.nO, obj.nP, obj.nI)),
    b=Biases("Bias vector", lambda obj: (obj.nO, obj.nP)),
    pad=Synapses(
        "Pad",
        lambda obj: (1, obj.nF, obj.nO, obj.nP),
        lambda M, ops: ops.normal_init(M, 1),
    ),
    d_W=Gradient("W"),
    d_pad=Gradient("pad"),
    d_b=Gradient("b"),
)
class PrecomputableAffine(Model):
    def __init__(
        self,
        nO: Optional[Any] = ...,
        nI: Optional[Any] = ...,
        nF: Optional[Any] = ...,
        nP: Optional[Any] = ...,
        **kwargs
    ):
        self.nO = ...
        self.nP = ...
        self.nI = ...
        self.nF = ...
    def begin_update(self, X, drop=...): ...
    def _add_padding(self, Yf): ...
    def _backprop_padding(self, dY, ids): ...
    @staticmethod
    def init_weights(model):
        """This is like the 'layer sequential unit variance', but instead
        of taking the actual inputs, we randomly generate whitened data.

        Why's this all so complicated? We have a huge number of inputs,
        and the maxout unit makes guessing the dynamics tricky. Instead
        we set the maxout weights to values that empirically result in
        whitened outputs given whitened inputs.
        """
        ...

def link_vectors_to_models(vocab): ...
def PyTorchBiLSTM(nO, nI, depth, dropout=...): ...
def Tok2Vec(width, embed_size, **kwargs): ...
def reapply(layer, n_times): ...
def asarray(ops, dtype): ...
def _divide_array(X, size): ...
def get_col(idx): ...
def doc2feats(cols: Optional[Any] = ...): ...
def print_shape(prefix): ...
@layerize
def get_token_vectors(tokens_attrs_vectors, drop=...): ...
@layerize
def logistic(X, drop=...): ...
def zero_init(model): ...
def getitem(i): ...
@describe.attributes(
    W=Synapses("Weights matrix", lambda obj: (obj.nO, obj.nI), lambda W, ops: None)
)
class MultiSoftmax(Affine):
    """Neural network layer that predicts several multi-class attributes at once.
    For instance, we might predict one class with 6 variables, and another with 5.
    We predict the 11 neurons required for this, and then softmax them such
    that columns 0-6 make a probability distribution and coumns 6-11 make another.
    """

    name = ...
    def __init__(self, out_sizes, nI: Optional[Any] = ..., **kwargs):
        self.out_sizes = ...
        self.nO = ...
        self.nI = ...
    def predict(self, input__BI): ...
    def begin_update(self, input__BI, drop=...): ...

def build_tagger_model(nr_class, **cfg): ...
def build_morphologizer_model(class_nums, **cfg): ...
@layerize
def SpacyVectors(docs, drop=...): ...
def build_text_classifier(nr_class, width=..., **cfg): ...
def build_bow_text_classifier(
    nr_class,
    ngram_size=...,
    exclusive_classes: bool = ...,
    no_output_layer: bool = ...,
    **cfg
): ...
@layerize
def cpu_softmax(X, drop=...): ...
def build_simple_cnn_text_classifier(
    tok2vec, nr_class, exclusive_classes: bool = ..., **cfg
):
    """
    Build a simple CNN text classifier, given a token-to-vector model as inputs.
    If exclusive_classes=True, a softmax non-linearity is applied, so that the
    outputs sum to 1. If exclusive_classes=False, a logistic non-linearity
    is applied instead, so that outputs are in the range [0, 1].
    """
    ...

def build_nel_encoder(embed_width, hidden_width, ner_types, **cfg): ...
@layerize
def flatten(seqs, drop=...): ...
def concatenate_lists(*layers, **kwargs):
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`
    """
    ...

def masked_language_model(vocab, model, mask_prob=...):
    """Convert a model into a BERT-style masked language model"""
    ...

class _RandomWords(object):
    def __init__(self, vocab):
        self.words = ...
        self.probs = ...
        self.words = ...
        self.probs = ...
        self.probs = ...
    def next(self): ...

def _apply_mask(docs, random_words, mask_prob=...): ...
def _replace_word(word, random_words, mask=...): ...
def _uniform_init(lo, hi): ...
@describe.attributes(
    nM=Dimension("Vector dimensions"),
    nC=Dimension("Number of characters per word"),
    vectors=Synapses(
        "Embed matrix", lambda obj: (obj.nC, obj.nV, obj.nM), _uniform_init(-0.1, 0.1)
    ),
    d_vectors=Gradient("vectors"),
)
class CharacterEmbed(Model):
    def __init__(self, nM: Optional[Any] = ..., nC: Optional[Any] = ..., **kwargs):
        self.nM = ...
        self.nC = ...
    @property
    def nO(self): ...
    @property
    def nV(self): ...
    def begin_update(self, docs, drop=...): ...

def get_cossim_loss(yh, y, ignore_zeros: bool = ...): ...