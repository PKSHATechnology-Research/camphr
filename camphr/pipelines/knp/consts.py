from typing import NamedTuple


class KnpUserKeyType(NamedTuple):
    element: str  # pyknp.Bunsetsu, Tag, Morpheme
    spans: str  # span list containing Bunsetsu, Tag, Morpheme correspondings.
    list_: str  # list containing knp elements
    parent: str
    children: str


class KnpUserKeys(NamedTuple):
    tag: KnpUserKeyType
    bunsetsu: KnpUserKeyType
    morph: KnpUserKeyType


KNP_USER_KEYS = KnpUserKeys(
    *[
        KnpUserKeyType(
            *["knp_" + comp + f"_{type_}" for type_ in KnpUserKeyType._fields]
        )
        for comp in KnpUserKeys._fields
    ]
)
