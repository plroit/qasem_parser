import torch
from typing import List, Tuple

from qanom.question_info import get_slots, get_role

from .common_defs import ArgInputExample, iter_batches, QasemFrame, QasemArgument, TokenizedSentence

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerBase

from .torch_utils import get_device


def find_answer_idx_with_fallback(tokens: List[str], answer: str):
    # try to locate the answer in the original text
    answer_start, answer_end = find_answer_idx(tokens, answer)
    if answer_start is None:
        # try case insensitive
        answer_start, answer_end = find_answer_idx(
            [tok.lower() for tok in tokens], answer.lower()
        )
    return answer_start, answer_end


def find_answer_idx(tokens: List[str], answer: str):
    # TODO: return closest index next to the predicate, not first index in the sentence
    # If the text token is in the prefix of the answer
    possible_starts = [idx for idx, tok in enumerate(tokens)
                       if answer[:len(tok)] == tok]
    for first_token_idx in possible_starts:
        end_token_idx = find_answer_from_token(tokens, first_token_idx, answer)
        if end_token_idx is not None:
            return first_token_idx, end_token_idx,
    return None, None


def find_answer_from_token(tokens: List[str], start_token_idx: int, answer: str):
    """A fuzzy matching between tokenized text and some answer string.

    The matching ignores differences in tokenization spaces, such that
    it is still able to match if the answer misses some spaces that exist
    in the tokenized text.

    example:
    original text: On Friday , Clark posted to Facebook
    tokens: ["On", "Friday", ",", "Clark", "posted", "to", "Facebook"]
    answer: "On Friday, Clark posted to Facebook"
    """
    curr_answer_char = 0
    search_tokens = tokens[start_token_idx:]
    answer = answer.strip()
    if not answer:
        return None

    for curr_text_idx, token in enumerate(search_tokens):
        # assume:
        # curr_answer_char is at a new non-space character
        # and we haven't matched the full answer yet
        assert(not str.isspace(answer[curr_answer_char]))
        assert(curr_answer_char < len(answer))

        n_chars = len(token)
        end_char_idx = curr_answer_char + n_chars
        token_in_answer = answer[curr_answer_char: end_char_idx]

        # no match, return
        if token != token_in_answer:
            return None
        curr_answer_char += n_chars

        # skip whitespace
        while curr_answer_char < len(answer):
            is_space = str.isspace(answer[curr_answer_char])
            if not is_space:
                break
            curr_answer_char += 1
            continue

        if curr_answer_char == len(answer):
            return start_token_idx + curr_text_idx + 1
    return None


_DEFAULT_BATCH_SIZE = 32
_DEFAULT_NUM_BEAMS = 4
_DEFAULT_MAX_LENGTH = 256


class T2TQasemArgumentParser:

    _PREDICATE_START_TOKEN = "<extra_id_0>"
    _PREDICATE_END_TOKEN = "<extra_id_1>"
    _QA_SEPARATOR = "<extra_id_2>"
    _PARSE_PREFIX_TOKENS = ["Generate",  "QA",  "pairs:"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 batch_size=_DEFAULT_BATCH_SIZE,
                 num_beams=_DEFAULT_NUM_BEAMS,
                 max_length=_DEFAULT_MAX_LENGTH
    ):
        """

        :param model: The encoder-decoder model to use.
        :param tokenizer: An instance of the tokenizer to use
        :param batch_size: The number of examples to process concurrently in a batch.
        :param num_beams: The number of beams in beam-search to use in decoding.
        :param max_length: Maximum length of the generated output Q&A pairs.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @staticmethod
    def hack_unknown_tokens(tokens: List[str]):
        # The default T5 tokenizer cannot handle the ` token that is often found
        # in quotes in some types of preprocessed text and academic datasets.
        # Instead it replaces it with an <UNK> token.
        # Since this case is relatively common, we handle it here.
        hacked_tokens = [
            token if token != "``" else '"'
            for token in tokens
        ]
        return hacked_tokens

    @classmethod
    def from_pretrained(cls, path_or_model_name: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path_or_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_or_model_name)
        device = get_device(**kwargs)
        model = model.to(device)
        return cls(model, tokenizer, **kwargs)

    @staticmethod
    def _prepare_prompt(sample: ArgInputExample):
        # prompt:
        # "Generate QA pairs: The fox <extra_id_0> jumped <extra_id_0> over the fence"
        tokens = sample.sentence
        predicate_index = sample.predicate.index
        # hacked_tokens = T2TQasemArgumentParser.hack_unknown_tokens(tokens)

        # Prefix that starts the prompt (our T5-model was trained with this prefix)
        new_tokens = T2TQasemArgumentParser._PARSE_PREFIX_TOKENS[:]
        # take the sentence till the predicate token
        new_tokens.extend(tokens[:])
        # put the predicate token between two special tokens marking start and end.
        # The model is trained without spaces between the special tokens and the predicate
        marked_predicate = "".join([T2TQasemArgumentParser._PREDICATE_START_TOKEN,
                                    tokens[predicate_index],
                                    T2TQasemArgumentParser._PREDICATE_END_TOKEN])
        new_tokens.append(marked_predicate)
        # put the rest of the sentence
        new_tokens.extend(tokens[(predicate_index + 1):])
        # voila, your prompt is ready
        return " ".join(new_tokens)

    def _prepare_batch(self, batch):
        inputs = [self._prepare_prompt(item) for item in batch]
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        return inputs

    def _predict_single_batch(self, batch: List[ArgInputExample]) -> List[List[QasemArgument]]:
        inputs = self._prepare_batch(batch)
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(inputs['input_ids'],
                                      num_beams=self.num_beams,
                                      max_length=self.max_length)
        outputs = outputs.detach().cpu()
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        post_processed = [
            self._postprocess(dec, item.sentence)
            for dec, item in zip(decoded, batch)
        ]
        return post_processed

    def _parse_question(self, question: str) -> Tuple[str, str]:
        # slots = get_slots(question,
        #                   append_is_negated_slot=True,
        #                   append_is_passive_slot=True,
        #                   append_verb_slot_inflection=True)
        role = get_role(question)
        if role:
            # let's not use qanom.SemanticRole enum
            # it is coupled with prepositions in a specific dataset
            # instead of representing the role as a syntactic position
            # such as R0, R1, R2 or an adjunct and an optional preposition
            role = role.name
        clean_question = question.replace("_", "")
        toks = [t.strip() for t in clean_question.split() if t.strip()]
        if toks[-1] == "?":
            clean_question = " ".join(toks[:-1]) + "?"
        return clean_question, role

    def _postprocess(self, decoded: str, tokens: TokenizedSentence):
        """
        Processes the generated output by a Text-to-Text model
        and parses it into a set of semantic arguments and their questions.
        :param decoded: The decoded output string produced by beam-search or other decoding algorithm.
        The correct format of the output is:
        <question_1>?answer_1;answer_2<extra_id_2><question_2>?<answer_1>...
        :param tokens: The original, pre-tokenized, sentence tokens
        :return:
        """
        arguments = []

        # we did not skip special tokens because we rely on <extra_id_2>
        # now need to remove other special tokens by ourselves
        decoded2 = decoded.replace(
            self.tokenizer.pad_token, "").replace(
            self.tokenizer.eos_token, "").strip(
        )
        qa_pairs = decoded2.split(self._QA_SEPARATOR)
        for raw_qa_pair in qa_pairs:
            qa_splits = raw_qa_pair.split("?", maxsplit=1)
            if len(qa_splits) <= 1:
                continue
            question = qa_splits[0].strip() + "?"
            question, role = self._parse_question(question)
            # this is the not a good choice since
            # a ";" sign may be part of an answer..
            # but that's how the model was trained :-(
            answers = qa_splits[1].split(";")
            answers = [ans.strip() for ans in answers]
            for answer in answers:
                # try to locate the answer in the original text
                answer_start, answer_end = find_answer_idx_with_fallback(tokens, answer)
                if answer_start is None:
                    continue
                arg_text = " ".join(tokens[answer_start: answer_end])
                arg = QasemArgument(arg_text, question, answer_start, answer_end, role)
                arguments.append(arg)

        return arguments

    def predict(self, items: List[ArgInputExample]) -> List[QasemFrame]:
        if not isinstance(items[0].sentence, List):
            raise ValueError("Sentences must be tokenized (list of tokens per sentence) when used with the ArgumentParser")

        all_qasem_arg_lists = []
        with torch.no_grad():
            for batch in iter_batches(items, self.batch_size, desc="Running argument parser"):
                post_processed = self._predict_single_batch(batch)
                all_qasem_arg_lists.extend(post_processed)
        return [
            QasemFrame(inp_item.sentence, inp_item.predicate, qasem_args)
            for inp_item, qasem_args in zip(items, all_qasem_arg_lists)
        ]