from spacy_pytorch_transformers.wordpiecer import PyTT_WordPiecer
from spacy_pytorch_transformers.util import get_sents
from spacy_pytorch_transformers._tokenizers import SerializableBertTokenizer


class BertWordPiecer(PyTT_WordPiecer):
    def predict(self, docs):
        """Run the word-piece tokenizer on a batch of docs and return the
        extracted strings.

        docs (iterable): A batch of Docs to process.
        RETURNS (tuple): A (strings, None) tuple.
        """
        output = []
        for doc in docs:
            doc_words = []
            doc_align = []
            offset = 0
            for sent in get_sents(doc):
                sent_words = []
                sent_align = []
                for segment in sent._.pytt_segments:
                    seg_words = self.model.tokenize(
                        " ".join(map(lambda x: x.text, segment))
                    )
                    print(" ".join(map(lambda x: x.text, segment)))
                    seg_words, seg_align = self._align(
                        segment, seg_words, offset=offset
                    )
                    assert len(segment) == len(seg_align)
                    sent_words.append(seg_words)
                    sent_align.append(seg_align)
                sw_flat = self.model.add_special_tokens(sent_words)
                sa_flat = self.model.fix_alignment(sent_align)
                doc_words.extend(sw_flat)
                doc_align.extend(sa_flat)
                offset += len(sw_flat)
            output.append((doc_words, doc_align))
        return output
