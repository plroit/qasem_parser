from qasem_parser import QasemParser, ArgInputExample, T2TQasemArgumentParser, T2TPropBankArgumentParser
from qasem_parser import Predicate

untokenized_sentences = [
    "The fox jumped over the fence.",
    "Back in May, a signal consistent with that of a radio beacon was detected in the area, but nothing turned up that helped with the search."
]

tokenized_sentences = [
    "Unfortunately , extensive property damage is bound to occur even with the best preparation .".split(),
    "Plans had originally called for the new President , or President-elect , of "
    "Afghanistan to attend the summit after following the election in June , "
    "but the ongoing audit of votes has made this impossible .".split(),
    "Dismal sales at General Motors Corp. dragged the U.S. car and truck market down below year - ago levels in early October , the first sales period of the 1990 model year .".split()
]


# test_sentence = "An inquest into the death of a man who died of measles has been opened and adjourned after a post-mortem examination failed to establish how he got the illness ."


# # joint_parser_path = "/home/nlp/plroit/pretrained/qasem/flan_t5_large_joint"
# joint_parser_path = "/home/nlp/ariecattan/qasem/qa_generation/models/sentence2list/question_answer/flan-t5-large_joint_tokenized_epoch_5"
# qasem_parser = QasemParser.from_pretrained(joint_parser_path)

# predicates = qasem_parser.predicate_detector.predict([test_sentence.split()])
# print(predicates)

# # Simple use case, parse the sentences end-to-end:
# frames = qasem_parser(untokenized_sentences)
# for sentence, frames_per_sent in zip(untokenized_sentences, frames):
#     print(sentence)
#     predicates = [frame.predicate.lemma for frame in frames_per_sent]
#     print(f"Predicates: {predicates}")

# frames = qasem_parser(tokenized_sentences, is_pretokenized=True)
# for sentence, frames_per_sent in zip(tokenized_sentences, frames):
#     print(sentence)
#     predicates = [frame.predicate.lemma for frame in frames_per_sent]
#     print(f"Predicates: {predicates}")

# Advanced use case, we know the predicate index in advance and have tokenized our sentence:
# arg_parser = T2TQasemArgumentParser.from_pretrained(joint_parser_path)
# frames = arg_parser(examples)
examples = [
    ArgInputExample(tokenized_sentences[2], Predicate("sale", "sales", 1, "Noun")),
    ArgInputExample(tokenized_sentences[2], Predicate("drag", "dragged", 6, "Verb")),
    ArgInputExample(tokenized_sentences[2], Predicate("sale", "sales", 25, "Verb")),

]
# 
pb_path = "/home/nlp/ariecattan/qasem/qa_generation/models/sentence2list/question_answer/flan-t5-large_ontonotes/checkpoint-14000"
arg_parser = T2TPropBankArgumentParser.from_pretrained(pb_path)
# arg_parser = T2TQasemArgumentParser.from_pretrained(joint_parser_path)
frames = arg_parser(examples)
print(frames)
