import itertools
from typing import List, Union

from spacy.tokens import Doc
from tqdm import tqdm

from dataclasses import dataclass

from abc import ABC

TokenizedSentence = List[str]
UntokenizedSentence = str


@dataclass(frozen=True)
class Predicate:
    lemma: str
    text: str
    index: int
    pos: str


@dataclass(frozen=True)
class QasemArgument:
    text: str
    question: str
    start_token: int
    end_token: int


@dataclass
class ArgInputExample:
    sentence: TokenizedSentence
    predicate: Predicate


@dataclass
class QasemFrame:
    sentence: TokenizedSentence
    predicate: Predicate
    arguments: List[QasemArgument]


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
