from qasem_parser import QasemParser, ArgInputExample, T2TQasemArgumentParser
from qasem_parser import Predicate

untokenized_sentences = [
    "The fox jumped over the fence.",
    "Back in May, a signal consistent with that of a radio beacon was detected in the area, but nothing turned up that helped with the search."
]

tokenized_sentences = [
    "Unfortunately , extensive property damage is bound to occur even with the best preparation .".split(),
    "Plans had originally called for the new President , or President-elect , of "
    "Afghanistan to attend the summit after following the election in June , "
    "but the ongoing audit of votes has made this impossible .".split()
]

joint_parser_path = "cattana/flan-t5-large-qasem-joint-tokenized"
qasem_parser = QasemParser.from_pretrained(joint_parser_path)

# Simple use case, parse the sentences end-to-end:
frames = qasem_parser(untokenized_sentences)
for sentence, frames_per_sent in zip(untokenized_sentences, frames):
    print(sentence)
    for frame in frames_per_sent:
        print(frame)

frames = qasem_parser(tokenized_sentences, is_pretokenized=True)
for sentence, frames_per_sent in zip(tokenized_sentences, frames):
    print(sentence)
    for frame in frames_per_sent:
        print(frame)
print()
print()
# Advanced use case, we know the predicate index in advance and have tokenized our sentence:
examples = [ArgInputExample(tokenized_sentences[0], Predicate("damage", "damage", 4, "Noun"))]
arg_parser = T2TQasemArgumentParser.from_pretrained(joint_parser_path)
frames = arg_parser(examples)
print(frames[0])