from spacy.pipeline.entityruler import EntityRuler
from spacy.language import Language
import bedoner.ner_labels.labels_ontonotes as L

person_patterns = [
    {"label": L.PERSON, "pattern": [{"TAG": "名詞,固有名詞,人名,姓"}, {"TAG": "名詞,固有名詞,人名,名"}]},
    {"label": L.PERSON, "pattern": [{"TAG": "名詞,固有名詞,人名,姓"}]},
    {"label": L.PERSON, "pattern": [{"TAG": "名詞,固有名詞,人名,名"}]},
]


def create_person_ruler(nlp: Language) -> EntityRuler:
    ruler = EntityRuler(nlp)
    ruler.add_patterns(person_patterns)
    return ruler
