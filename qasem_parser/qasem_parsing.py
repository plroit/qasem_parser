
import spacy
from spacy.tokens import Doc
from typing import List, Union

from . import T2TQasemArgumentParser, BertPredicateDetector
from .common_defs import ArgInputExample, TokenizedSentence, UntokenizedSentence, \
    QasemFrame, PredicateDetector, ArgumentParser, Predicate

from tqdm import tqdm

from .ling_utils import spacy_analyze


_DEFAULT_NOMINAL_DETECTOR = "kleinay/nominalization-candidate-classifier"

# This is the old parser, it was trained with a different format
# for questions and answers, don't use with the current version of code
# _DEFAULT_JOINT_ARG_PARSER = "kleinay/qanom-seq2seq-model-joint"


def create_arg_input_sample(docs: List[Doc], predicates: List[List[Predicate]]) -> ArgInputExample:
    return [
        ArgInputExample([t.text for t in doc], predicate)
        for doc, doc_preds in zip(docs, predicates)
        for predicate in doc_preds
    ]


def _group_by_sentences(frames, predicates):
    # let's group back the frames according to sentences,
    # s.t. the ith list of frames corresponds to the ith sentence.
    res = []
    idx = 0
    frame_counts = [len(doc_preds) for doc_preds in predicates]
    for frame_count in frame_counts:
        end_idx = idx + frame_count
        doc_frames = frames[idx: end_idx]
        res.append(doc_frames)
        idx = end_idx
    return res


class QasemParser:

    @classmethod
    def from_pretrained(cls,
                        arg_parser_path: str,
                        nom_predicate_detector_path=_DEFAULT_NOMINAL_DETECTOR,
                        spacy_lang="en_core_web_sm"
    ):
        # TODO: make this more generic? how to initialize the correct parser class just from the model path?
        nlp = spacy.load(spacy_lang)
        arg_parser = T2TQasemArgumentParser.from_pretrained(arg_parser_path)
        predicate_detector = BertPredicateDetector.from_pretrained(nom_predicate_detector_path, nlp)
        return cls(arg_parser, predicate_detector, nlp)

    def __init__(self, arg_parser: ArgumentParser, predicate_detector: PredicateDetector, spacy_lang: spacy.Language):
        self.arg_parser = arg_parser
        self.predicate_detector = predicate_detector
        self._nlp = spacy_lang

    def _normalize_input(self, sentences, is_pretokenized: bool) -> List[Doc]:
        if not sentences:
            raise ValueError("sentences")
        out_sentences = sentences
        is_str = isinstance(sentences, str)
        is_list_of_str = isinstance(sentences, list) and isinstance(sentences[0], str)
        is_list_of_list = isinstance(sentences, list) and isinstance(sentences[0], list)
        if not is_pretokenized:
            # in the untokenized case a single string is a sentence.
            if is_str:
                out_sentences = [sentences]
            elif is_list_of_list:
                # can't have List[List[str]] in the untokenized case
                raise ValueError("pretokenized=False, sentences must be either "
                                 "a list of strings each representing a sentence "
                                 "or a single string for a single sentence ")
            # spacy will tokenize and analyze the sentences
            docs = list(tqdm(self._nlp.pipe(out_sentences),
                             desc="Running spacy for initial tokenization",
                             total=len(out_sentences)))
        else:
            # pretokenized=True
            if is_str:
                raise ValueError(
                    "pretokenized=True, sentences must be a list of "
                    "tokens or a batch of lists of tokens")
            # if it is a single pre-tokenized example
            if is_list_of_str:
                out_sentences = [sentences]
            docs = spacy_analyze(out_sentences, self._nlp)
        return docs

    def __call__(self,
                 sentences: Union[TokenizedSentence,
                                  UntokenizedSentence,
                                  List[TokenizedSentence],
                                  List[UntokenizedSentence]],
                 is_pretokenized=False) -> List[List[QasemFrame]]:

        # after normalization, sentences is a batch of tokenized sentences.
        docs = self._normalize_input(sentences, is_pretokenized)
        predicates = self.predicate_detector(docs)
        arg_input_samples = create_arg_input_sample(docs, predicates)
        frames = self.arg_parser(arg_input_samples)
        res = _group_by_sentences(frames, predicates)
        return res
