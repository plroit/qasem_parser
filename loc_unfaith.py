import os 
import pandas as pd 
from tqdm import tqdm 
import spacy 
from itertools import chain
import collections 
import json 
from spacy.tokens import Doc
import jsonlines
from qasem_parser import QasemParser, QasemFrame, QasemArgument
import argparse 
from difflib import SequenceMatcher

IMPORTANT_POS = ['ADJ', 'ADV', 'NOUN', 'NUM', 'PROPN', 'VERB', 'PRON']
STR_RESEMBLANCE_THR = 0.8

def flatten_list(lst):
  return list(chain.from_iterable(lst))

class LocUnfaith:
  def __init__(self, summary_id, summary_sentences) -> None:
    self.summary_id = summary_id
    self.df_sentences = summary_sentences
  
  def is_aligned(self, t1, t2):
    return SequenceMatcher(None, t1.lower(), t2.lower()).ratio() > STR_RESEMBLANCE_THR

  def _get_source_alignment(self, source_tokens, span):
    aligned_tokens = set()
    
    all_source_aligned_tokens = set([i for i, src_tok in enumerate(source_tokens) 
                                 if any([self.is_aligned(src_tok.lemma_, sum_tok) for sum_tok in span])])
    for src_tok in all_source_aligned_tokens:
      # either the source token is an important POS or there is an aligned token in a window of 2 with an important token
      if source_tokens[src_tok].pos_ in IMPORTANT_POS or \
          (src_tok+1 in all_source_aligned_tokens and source_tokens[src_tok+1].pos_ in IMPORTANT_POS) or \
          (src_tok-1 in all_source_aligned_tokens and source_tokens[src_tok-1].pos_ in IMPORTANT_POS):
        aligned_tokens.add(src_tok)

    return list(aligned_tokens)
    
  def _extract_qas_from_sentence(self, sentence_id, row):
    qas = []
    for predicate_frame in row["qa_frames"]:
      # update start and end tokens of predicate at the summary level
      predicate_token_index = predicate_frame.predicate.index + row["start_sentence_token"] 

      for frame in predicate_frame.arguments:
        # remove non-informative POS in the answer 
        # this enables to merge nodes in the graph (e.g., in Paris vs. Paris)
        # and show only the clean span in the first annotation step
        clean_start_answer = frame.start_token 
        while clean_start_answer <= frame.end_token and row["sentences"][clean_start_answer].pos_ not in IMPORTANT_POS:
          clean_start_answer += 1
        if clean_start_answer > frame.end_token:
          print(f"The answer {frame.text} is not a valid answer")
          continue
        
        # update start and end tokens of answers at the summary level
        clean_start_answer += row["start_sentence_token"]
        answer_start_token_index = frame.start_token + row["start_sentence_token"] 
        answer_end_token_index = frame.end_token + row["start_sentence_token"]

        # get source alignment to the answer
        lemma_span = [x.lemma_ for x in row["sentences"][frame.start_token:frame.end_token]]
        answer_source_alignment = self._get_source_alignment(row["spacy_source"], lemma_span)

        qas.append({
          "sentId": sentence_id,
          "predicateId": f'{predicate_token_index}-{predicate_token_index + 1}',
          "predicate": predicate_frame.predicate.text,
          "predicatePos": predicate_frame.predicate.pos,
          "question": frame.question,
          "answer": frame.text,
          "answerStartToken": [answer_start_token_index],
          "answerEndToken": [answer_end_token_index],
          "answerId": f'{answer_start_token_index}-{answer_end_token_index}',
          "cleanAnswerId": f'{clean_start_answer}-{answer_end_token_index}',
          "verbTokenId": frame.verb_token_id,
          "sourceIds": answer_source_alignment
        })

    return qas     
  
  
  def extract_qas_from_summary(self):
    qas = flatten_list(
      [self._extract_qas_from_sentence(i, row) for i, row in self.df_sentences.iterrows()]
    )
    for i, _ in enumerate(qas):
      qas[i]["questionId"] = i 
    return qas
  

  def extract_all_spans(self, qas):
    spans = []

    '''
    predicates
    answers
    '''
    predicates = collections.defaultdict(list)
    answers = collections.defaultdict(list)
    predicate_tokens = set() # to check afterwards whether spans include predicate

    df_qas = pd.DataFrame(qas)

    # create mapping from predicate_id and answer_id to question_ids they are involved
    for predicate_id, predicate_qas in df_qas.groupby("predicateId"):
      predicate_tokens.add(int(predicate_id.split("-")[0]))
      predicates[predicate_id] = predicate_qas["questionId"].tolist()
      for i, qa in predicate_qas.iterrows():
        answer_id = qa["cleanAnswerId"] # use cleanAnswerId to merge different answers to the same span
        answers[answer_id].append(qa["questionId"])

    spans = []

    # adding predicates
    for predicate_id, qa_ids in predicates.items():
      predicate_start, predicate_end = predicate_id.split("-")
      spans.append({
        "start": int(predicate_start),
        "end": int(predicate_end),
        "qaIds": qa_ids,
        "predicate": True,
        "include_predicate": True
      })

    # adding answers
    for answer_id, qa_ids in answers.items():
      answer_start, answer_end = answer_id.split("-")
      answer_start, answer_end = int(answer_start), int(answer_end)
      answer_tokens = set(range(answer_start, answer_end))
      # check if one the answer token is a predicate
      include_predicate = len(answer_tokens.intersection(predicate_tokens)) > 0 
      spans.append({
        "start": answer_start,
        "end": answer_end,
        "qaIds": qa_ids,
        "predicate": False,
        "include_predicate": include_predicate
      })

    spans = sorted(spans, key=lambda x: x["start"])
    for i, span in enumerate(spans):
      spans[i]["id"] = i
      start, end = span["start"], span["end"]
      span_text = [x.lemma_ for x in self.df_sentences.iloc[0]["spacy_summary"][start:end]]
      source_alignment = self._get_source_alignment(self.df_sentences.iloc[0]["spacy_source"], span_text)
      spans[i]["sourceIds"] = source_alignment
      
    return spans 
  

  def get_source_tokens(self):
    return [{
      "id": i,
      "text": token.text,
      "lemma": token.lemma_
    } for i, token in enumerate(self.df_sentences.iloc[0]["spacy_source"])]
    
  def get_summary_tokens(self, spans):
    """
    For each token, add the span id and the corresponding class (token or mention)
    token is a standard token
    mention is a token that participates either as a predicate in an answer
    """
    tokens = []

    token2span = collections.defaultdict(list)
    for i, span in enumerate(spans):
      for token_id in range(span["start"], span["end"]):
        token2span[token_id].append(i)

    labels = [None] * len(self.df_sentences.iloc[0]["spacy_summary"])
    if "labels" in self.df_sentences.columns: # in CLIFF
      labels = self.df_sentences.iloc[0]["labels"]

    for i, token in enumerate(self.df_sentences.iloc[0]["spacy_summary"]):
      tokens.append({
        "id": i,
        "text": token.text,
        "lemma": token.lemma_,
        "spans": token2span[i],
        "class": "token" if len(token2span[i]) == 0 else "mention",
        "label": labels[i]
      })
    
    return tokens

  def export_summary_data(self):
    """
    Post process the output of qasem parser 
    and prepare JSON input files for loc-unfaith 
    """
    source = self.get_source_tokens()
    qas = self.extract_qas_from_summary()
    if len(qas) == 0:
      return {}
    
    spans = self.extract_all_spans(qas)
    summary = self.get_summary_tokens(spans)

    return {
      "source": source,
      "summary": summary,
      "spans": spans,
      "qas": qas,
      "datasource": self.df_sentences.iloc[0]["origin"],
      "summaryId": f'{self.df_sentences.iloc[0]["id"]}',
      "aggrefactId": self.summary_id,
      "label": int(self.df_sentences.iloc[0]["label"]),
      "dataset": str(self.df_sentences.iloc[0]["dataset"])
    }
  
if __name__ == "__main__":
  # load data
  data_path = "/home/nlp/ariecattan/summarization/factuality/cliff/cliff_raw.jsonl"
  nlp = spacy.load("en_core_web_sm")

  with jsonlines.open(data_path, "r") as f:
    data = [x for x in f]

  df = pd.DataFrame(data)
  df["summary_tokens"] = df["summary"].apply(lambda s: s.split())
  df["spacy_inputs"] = df["summary_tokens"].progress_apply(
    lambda tokens: Doc(words=tokens, vocab=nlp.vocab)) 
  
  docs = list(tqdm(nlp.pipe(list(df["spacy_inputs"])), desc="Running spacy on summary", total=len(df)))
  df["spacy_summary"] = docs
  df["sentences"] = df["spacy_summary"].apply(lambda x: [sent for sent in x.sents])
  df_sentences = df.explode("sentences")

  # add start sentence token to each sentence, in order to get summary-level token ids
  df_sentences["start_sentence_token"] = df_sentences["sentences"].apply(lambda x: x.start) 

  groups = df_sentences.groupby("id")
  summary_id = "0_bart_xsum"
  sentences = groups.get_group(summary_id)

  loc = LocUnfaith(summary_id, sentences)