from typing import List, Union

import spacy
import torch
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources
from spacy.tokens import Doc
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

from . import torch_utils
from .common_defs import PredicateDetector, TokenizedSentence, Predicate, iter_batches
from .ling_utils import spacy_analyze

_DEFAULT_BATCH_SIZE = 32
# token classification threshold for nominal predicates
_DEFAULT_NOMINAL_THRESHOLD = 0.75


class BertPredicateDetector(PredicateDetector):
    COMMON_NOUNS = {"NN", "NNS"}

    @classmethod
    def from_pretrained(
            cls,
            nominal_classifier_path: str,
            spacy_model_or_name: Union[str, spacy.Language] = 'en_core_web_sm',
            **kwargs
    ):
        device = torch_utils.get_device(**kwargs)
        if isinstance(spacy_model_or_name, str):
            nlp = spacy.load(spacy_model_or_name)
        else:
            nlp = spacy_model_or_name
        nom_tokenizer = AutoTokenizer.from_pretrained(nominal_classifier_path)
        nom_classifier = AutoModelForTokenClassification.from_pretrained(nominal_classifier_path)
        nom_classifier = nom_classifier.to(device)
        return cls(nom_classifier, nom_tokenizer, nlp, **kwargs)

    def __init__(self,
                 nom_model: PreTrainedModel,
                 nom_tokenizer: PreTrainedTokenizerBase,
                 spacy_lang: spacy.Language,
                 threshold=_DEFAULT_NOMINAL_THRESHOLD,
                 batch_size=_DEFAULT_BATCH_SIZE):
        self.nom_model = nom_model.eval()
        self.nom_tokenizer = nom_tokenizer
        self.nlp = spacy_lang
        self.batch_size = batch_size
        self.threshold = threshold
        # not sure why they store True as a string?
        self.positive_label_idx = nom_model.config.label2id['True']

    @staticmethod
    def _is_verbal_predicate(tok: spacy.tokens.Token):
        is_verb = tok.pos_ == "VERB"
        not_modal = tok.tag_ != "MD"
        not_be = tok.lemma_.lower() != "be"
        return is_verb and not_modal and not_be

    def detect_verbal(self, docs: List[Doc]) -> List[List[Predicate]]:
        predicates = [[] for _ in range(len(docs))]
        for sent_idx, doc in enumerate(docs):
            for tok in doc:
                if self._is_verbal_predicate(tok):
                    predicate = Predicate(tok.lemma_, tok.text, tok.i, tok.pos_)
                    predicates[sent_idx].append(predicate)
        return predicates

    def _detect_nominal_batch(self, batch: List[Doc]) -> List[List[Predicate]]:
        # the output for this batch of sentences
        predicates = [[] for _ in range(len(batch))]

        inputs = self._prepare_inputs(batch)
        inputs = inputs.to(self.nom_model.device)
        # forward call, let's get the logits
        logits = self.nom_model(**inputs).logits.detach().cpu()
        # while this model is for binary classification, for some reason
        # it was trained with two output logits.
        # we need to softmax them to get the prob right.
        probs = logits.softmax(axis=-1)
        positive_probs = probs[:, :, self.positive_label_idx]
        is_nominal_predicate = positive_probs > self.threshold
        batch_indices, seq_indices = is_nominal_predicate.nonzero(as_tuple=True)
        for batch_idx, seq_idx in zip(batch_indices, seq_indices):
            doc = batch[batch_idx]
            word_idx = inputs.token_to_word(batch_idx, seq_idx)
            pred_token = doc[word_idx]
            predicate = Predicate(pred_token.lemma_.lower(),
                                  pred_token.text,
                                  word_idx,
                                  pred_token.pos_)
            predicates[batch_idx].append(predicate)
        return predicates

    def transform_with_verb_forms(self, doc_preds: List[Predicate]) -> List[Predicate]:
        new_preds = []
        for pred in doc_preds:
            verb_forms, is_ok = get_verb_forms_from_lexical_resources(pred.lemma)
            if not is_ok:
                continue
            new_pred = Predicate(verb_forms[0], pred.text, pred.index, pred.pos)
            new_preds.append(new_pred)
        return new_preds
    
    def detect_nominal_spacy(self, docs: List[Doc]) -> List[List[Predicate]]:
        all_predicates: List[List[Predicate]] = []
        for doc in docs:
            predicate_docs = []
            for nn in doc._.nominalizations:
                predicate_docs.append(
                    Predicate(nn._.verb_form, nn.text, nn.index, nn.pos)
                )
            all_predicates.append(predicate_docs)

        return all_predicates


    def detect_nominal(self, docs: List[Doc]) -> List[List[Predicate]]:
        # predict using the nominal classifier which token ids
        # are nominal predicates
        all_predicates: List[List[Predicate]] = []
        with torch.no_grad():
            for batch in iter_batches(docs, batch_size=self.batch_size):
                predicates = self._detect_nominal_batch(batch)
                all_predicates.extend(predicates)

        # remove predicates which are not common nouns
        # This is probably a mistake by either spacy or the nominal classifier
        # My bet is on the nominal classifier though since it had probably less data to train on.
        all_predicates = [
            [pred for pred in doc_preds if doc[pred.index].tag_ in self.COMMON_NOUNS]
            for doc, doc_preds in zip(docs, all_predicates)
        ]
        # Use qanom to find a verbal form for nominal predicates
        # change the lemma of the predicate to its verb form
        all_predicates = [
            self.transform_with_verb_forms(doc_preds)
            for doc_preds in all_predicates
        ]
        return all_predicates

    def _prepare_inputs(self, docs: List[Doc]):
        texts = [[tok.text for tok in doc] for doc in docs]
        batch = self.nom_tokenizer(
            texts,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True
        )
        return batch

    def predict(self, sentences: List[Union[Doc, TokenizedSentence]]) -> List[List[Predicate]]:
        # let's syntactically analyze the sentences first:
        if not sentences:
            return []
        if isinstance(sentences[0], Doc):
            docs = sentences
        else:
            docs = spacy_analyze(sentences, self.nlp)
        verb_predicates = self.detect_verbal(docs)
        # nom_predicates = self.detect_nominal(docs)
        nom_predicates = self.detect_nominal_spacy(docs)
        all_predicates = [verb_preds + noun_preds
                          for verb_preds, noun_preds
                          in zip(verb_predicates, nom_predicates)]
        return all_predicates
