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
from loc_unfaith import LocUnfaith

tqdm.pandas()


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--data_path", 
    type=str, 
    help='path to jsonl file where each row includes at least'
      'the fields `summary`, `article`, `datasource` and `label` (or `labels` for CLIFF).',
    default="/home/nlp/ariecattan/summarization/factuality/cliff/cliff_raw.jsonl"
  )
  parser.add_argument(
    "--model_name_or_path", 
    type=str, 
    help="model name on HF on path",
    default="/home/nlp/ariecattan/qasem/qa_generation/models/sentence2list/question_answer/flan-t5-xl_joint_tokenized_epoch_5"
  )
  parser.add_argument(
    "--spacy_lang",
    type=str,
    help="name of spacy model",
    default="en_core_web_lg"
  )
  parser.add_argument(
    "--output_dir",
    type=str,
    help="directory to save json files"
  )
  parser.add_argument(
    "--dataset_name",
    type=str, 
    help="name of faithfulness dataset",
    default="cliff"
  )
  parser.add_argument(
    "--pretokenized",
    action="store_true",
    help="whether the summary is already pretokenized"
  )
  args = parser.parse_args()
  
  # load models
  nlp = spacy.load(args.spacy_lang)
  parser = QasemParser.from_pretrained(args.model_name_or_path, spacy_lang=args.spacy_lang)

  # load data
  with jsonlines.open(args.data_path, "r") as f:
    data = [x for x in f]

  df = pd.DataFrame(data)
  if args.dataset_name == "cliff":
    df["label"] = df["labels"].apply(
      lambda labels: True if sum(1 for x in labels if x == "correct") == len(labels) else False
    )
  df["origin"] = df["datasource"] 
  df["dataset"] = args.dataset_name
  
  

  # run spacy on source 
  spacy_docs_source = list(tqdm(nlp.pipe(df["article"]), 
                                desc='Running spacy on source', 
                                total=len(df)))
  df["spacy_source"] = spacy_docs_source


  # run spacy on summary
  if args.pretokenized:
    df["summary_tokens"] = df["summary"].apply(lambda s: s.split())
    df["spacy_inputs"] = df["summary_tokens"].progress_apply(
      lambda tokens: Doc(words=tokens, vocab=nlp.vocab)) 
  else:
    df["spacy_inputs"] = df["summary"]
  docs = list(tqdm(nlp.pipe(list(df["spacy_inputs"])), desc="Running spacy on summary", total=len(df)))
  df["spacy_summary"] = docs
    
  # split summary into sentences and create a dataframe at the sentence level
  df["sentences"] = df["spacy_summary"].apply(lambda x: [sent for sent in x.sents])
  df_sentences = df.explode("sentences")
  print(f'num sentences: {len(df_sentences)}')

  # add start sentence token to each sentence, in order to get summary-level token ids
  df_sentences["start_sentence_token"] = df_sentences["sentences"].apply(lambda x: x.start) 
  # this is a hack for re-running spacy on tokenized sentences, while reseting the index
  df_sentences["input_for_qasem"] = df_sentences["sentences"].apply(lambda sent: [token.text for token in sent]) 
  
  # run qasem parser 
  frames = parser(df_sentences["input_for_qasem"].tolist())
  df_sentences["qa_frames"] = frames   

  # create json file for each summary
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
  all_data = {}
  for summary_id, summary_sentences in tqdm(df_sentences.groupby("id"),
                                            total=len(df),
                                            desc="Creating loc-unfaith input files"):
    loc_unfaith = LocUnfaith(summary_id, summary_sentences.reset_index()) # reset index to reset sentence id 
    json_input_file = loc_unfaith.export_summary_data()
    path = f"{args.output_dir}/{summary_id}.json"
    all_data[summary_id] = json_input_file
    with open(path, "w") as f:
      json.dump(json_input_file, f, indent=4)