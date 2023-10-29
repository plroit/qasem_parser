from typing import List

import spacy
from tqdm import tqdm
from spacy.tokens import Doc
from .common_defs import TokenizedSentence


def spacy_analyze(sentences: List[TokenizedSentence], nlp: spacy.Language, verbose=True):
    # no need to count everything to determine if should disable if this is a huge payload anyway.
    num_tokens = sum(len(tokens) for tokens in sentences[:500])
    # the default is not to disable, however if payload is small disable anyway.
    disable_print = (not verbose) or (num_tokens < 500)
    docs = [Doc(words=tokens, vocab=nlp.vocab) for tokens in sentences]
    docs = list(tqdm(nlp.pipe(docs), disable=disable_print, desc="Running spacy...", total=len(sentences)))
    return docs
