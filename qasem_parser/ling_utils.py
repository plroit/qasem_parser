from typing import List

import spacy
from tqdm import tqdm
from spacy.tokens import Doc
from .common_defs import TokenizedSentence


def spacy_analyze(sentences: List[TokenizedSentence], nlp: spacy.Language, verbose=True):
    docs = [Doc(words=tokens, vocab=nlp.vocab) for tokens in sentences]
    docs = list(tqdm(nlp.pipe(docs), disable=not verbose, desc="Running spacy...", total=len(sentences)))
    return docs
