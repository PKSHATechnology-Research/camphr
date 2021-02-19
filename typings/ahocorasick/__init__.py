from typing import Iterator, Optional, Tuple


class Automaton:
    def iter(
        self, string: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> Iterator[Tuple[int, str]]:
        ...

    def make_automaton(self) -> None:
        ...

    def add_word(self, key: str, value: Optional[str] = None) -> bool:
        ...
