import inspect

import spacy
from spacy.tokens import Doc
from typing import List, Union

from . import T2TQasemArgumentParser, BertPredicateDetector, T2TPropBankArgumentParser
from .common_defs import ArgInputExample, TokenizedSentence, UntokenizedSentence, \
    QasemFrame, PredicateDetector, ArgumentParser, Predicate

from tqdm import tqdm

from .ling_utils import spacy_analyze


_DEFAULT_NOMINAL_DETECTOR = "kleinay/nominalization-candidate-classifier"
_DEFAULT_JOINT_ARG_PARSER = "cattana/flan-t5-large-qasem-joint-tokenized"

# This is the old parser, it was trained with a different format
# for questions and answers, don't use with the current version of code
# _DEFAULT_JOINT_ARG_PARSER = "kleinay/qanom-seq2seq-model-joint"



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
                        spacy_lang="en_core_web_sm", **kwargs
    ):
        # TODO: make this more generic? how to initialize the correct parser class just from the model path?
        nlp = spacy.load(spacy_lang)

        # There's got to be a better way than this hack :-)
        if "onto" in arg_parser_path:
            parser_cls = T2TPropBankArgumentParser
        else:
            parser_cls = T2TQasemArgumentParser
        parser_params = list(inspect.signature(parser_cls).parameters)
        parser_kwargs = {
            k: kwargs.get(k) for k in dict(kwargs)
            if k in parser_params
        }
        classifier_params = list(inspect.signature(BertPredicateDetector).parameters)
        classifier_kwargs = {
            k: kwargs.get(k) for k in dict(kwargs)
            if k in classifier_params
        }

        arg_parser = parser_cls.from_pretrained(arg_parser_path, **parser_kwargs)
        predicate_detector = BertPredicateDetector.from_pretrained(nom_predicate_detector_path, nlp, **classifier_kwargs)
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
        if is_list_of_list:
            # this must be pre-tokenized
            is_pretokenized = True
        elif is_str:
            # this must be a single sentence, untokenized
            is_pretokenized = False
            out_sentences = [out_sentences]
        elif is_list_of_str and is_pretokenized:
            # this is a single pre_tokenized sentence
            out_sentences = [out_sentences]
        elif is_list_of_str and not is_pretokenized:
            # this is a list of untokenized sentences
            pass
        # these conditions must hold now:
        if is_pretokenized:
            assert isinstance(out_sentences[0], List)
        else:
            assert isinstance(out_sentences[0], str)

        if is_pretokenized:
            docs = spacy_analyze(out_sentences, self._nlp)
        else:
            # spacy will tokenize and analyze the sentences
            docs = list(tqdm(self._nlp.pipe(out_sentences),
                             desc="Running spacy for initial tokenization",
                             total=len(out_sentences),
                             disable=len(out_sentences) < 100))
        return docs

    def __call__(self,
                 sentences: Union[TokenizedSentence,
                                  UntokenizedSentence,
                                  List[TokenizedSentence],
                                  List[UntokenizedSentence]],
                 is_pretokenized=False) -> List[List[QasemFrame]]:
        """
        Semantically parses the sentences.
        For example:
        sentences = [
            'The fox jumped over the fence .'.split(),
            'Unfortunately , extensive property damage is bound to occur even with the best preparation .'.split()
        ]
        qasem_frames = qasem_parser(sentences)
        #
        # First sentence frames:
        # [jump-v:  The fox (who jumped?) | over the fence (where did someone jump?)]
        print(qasem_frames[0])
        #
        # Second sentence frames:
        # [
        #   bind-v:  extensive property damage (what is bound to do something?) | occur (what is something bound to do?) | occur even with the best preparation (what is something bound to do?)
        #   occur-v:  extensive property damage (what occurs?) | even with the best preparation (how does something occur?)
        #   damage-n:  extensive property (what is damaged?) | even with the best preparation (how is something damaged?)
        #   prepare-n:  extensive property damage (what is prepared?)
        # ]
        print(qasem_frames[1])

        :param sentences:
        :param is_pretokenized: Used to disambiguate between
        List[UntokenizedSentence] and TokenizedSentence
        Since both types are a list of strings.
        If is_tokenized=True then the input is handled as a single tokenized sentence
        where each string in the list is a single token (word).
        If is_tokenize=false then the input is handled as a list of untokenized sentence
        where each string in the list is a whole sentence.
        :return: A list with the same length as the input sentences.
        result[i] is a list of QASem frames that occur in the ith sentence.
        The list of frames may be empty if no predicate is detected in the ith sentence.
        """

        if not sentences:
            return []
        # after normalization, sentences is a batch of spacy Docs.
        docs = self._normalize_input(sentences, is_pretokenized)
        # predicates: the i-th entry corresponds to the list of predicates
        # of the i-th sentence.  
        predicates: list[list[Predicate]] = self.predicate_detector(docs)
        # flatten the list but be cautious about sentences without predicates
        all_samples = []
        for sent, sent_preds in zip(docs, predicates):
            sent_tokens = [t.text for t in sent]
            sent_samples = [ArgInputExample(sent_tokens, predicate) for predicate in sent_preds]
            all_samples.extend(sent_samples)
        frames = self.arg_parser(all_samples)
        # group back per sent [[frame, .. frame], ..[frame], ...[],..]
        res: list[list[QasemFrame]] = _group_by_sentences(frames, predicates)
        return res
