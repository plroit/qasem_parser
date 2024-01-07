import itertools
from typing import List, Union, Optional

from spacy.tokens import Doc
from tqdm import tqdm

from dataclasses import dataclass

from abc import ABC

TokenizedSentence = List[str]
UntokenizedSentence = str


def _clean_question(raw_question: str) -> str:
    raw = raw_question.replace("_", "")

    splits = [s.strip() for s in raw.split() if s.strip()]
    if splits[-1] == "?":
        splits = splits[:-1]
    clean_question = " ".join(splits)
    if not clean_question.endswith("?"):
        clean_question += "?"
    return clean_question


@dataclass(frozen=True)
class Predicate:
    lemma: str
    text: str
    index: int
    pos: str

    def __repr__(self):
        return f"{self.lemma}-{self.pos[0].lower()}"


@dataclass(frozen=True)
class QasemArgument:
    text: str
    raw_question: str
    start_token: int
    end_token: int
    role: Optional[str] = None

    @property
    def question(self) -> str:
        clean_question = _clean_question(self.raw_question)
        return clean_question
    


    def __repr__(self):
        # The fox (R0: who jumped)
        return f"{self.text} ({self.role or ''}{': ' if self.role else ''}{self.question})"


@dataclass
class ArgInputExample:
    sentence: TokenizedSentence
    predicate: Predicate


@dataclass
class QasemFrame:
    sentence: TokenizedSentence
    predicate: Predicate
    arguments: List[QasemArgument]

    def __repr__(self):
        return f"{self.predicate}:  {' | '.join(str(a) for a in self.arguments)}"


class PredicateDetector(ABC):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, sentences: List[Union[TokenizedSentence, Doc]]) -> List[List[Predicate]]:
        ...


class ArgumentParser(ABC):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, samples: List[ArgInputExample]) -> List[QasemFrame]:
        ...


def iter_batches(items, batch_size, desc=None, verbose=True):
    if len(items) <= batch_size:
        yield items
        return

    curr_idx = 0
    with tqdm(total=len(items), desc=desc, disable=not verbose) as progress_bar:
        while curr_idx < len(items):
            batch = list(itertools.islice(items, curr_idx, curr_idx + batch_size))
            yield batch
            progress_bar.update(len(batch))
            curr_idx += len(batch)
