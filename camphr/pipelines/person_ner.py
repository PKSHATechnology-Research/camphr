from spacy.language import Language
from spacy.pipeline.entityruler import EntityRuler

import camphr.ner_labels.labels_ontonotes as L

person_patterns = [
    {"label": L.PERSON, "pattern": [{"TAG": "名詞,固有名詞,人名,姓"}, {"TAG": "名詞,固有名詞,人名,名"}]},
    {"label": L.PERSON, "pattern": [{"TAG": "名詞,固有名詞,人名,姓"}]},
    {"label": L.PERSON, "pattern": [{"TAG": "名詞,固有名詞,人名,名"}]},
]

LABEL_PERSON = L.PERSON


def create_person_ruler(nlp: Language) -> EntityRuler:
    """Create entity ruler that extracts person name with regex.

    Notes:
        This component must be used with `camphr.lang.mecab.Japanese`.
        In order to imporove accuracy, it is recommended to create a Mecab user dictionaly. See scripts/mecab_person_dictionary.

    Examples:
        >>> nlp = camphr.lang.mecab.Japanes()
        >>> nlp.add_pipe(create_person_ruler(nlp))
        >>> doc = nlp("今日は田中と散歩に行った")
        >>> doc.ents
        (田中, )
    """
    ruler = EntityRuler(nlp)
    ruler.add_patterns(person_patterns)
    return ruler
