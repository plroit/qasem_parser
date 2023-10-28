import torch
from typing import List
from .common_defs import ArgInputExample, iter_batches, QasemFrame, QasemArgument, TokenizedSentence

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerBase

from .torch_utils import get_device


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
        device = get_device(kwargs)
        model = model.to(device)
        return cls(model, tokenizer, **kwargs)

    @staticmethod
    def _prepare_prompt(sample: ArgInputExample):
        # prompt:
        # "Generate QA pairs: The fox <extra_id_0> jumped <extra_id_0> over the fence"
        tokens = sample.sentence
        predicate_index = sample.predicate.index
        hacked_tokens = T2TQasemArgumentParser.hack_unknown_tokens(tokens)

        # Prefix that starts the prompt (our T5-model was trained with this prefix)
        new_tokens = T2TQasemArgumentParser._PARSE_PREFIX_TOKENS[:]
        # take the sentence till the predicate token
        new_tokens.extend(hacked_tokens[:])
        # put the predicate token between two special tokens marking start and end.
        new_tokens.append(T2TQasemArgumentParser._PREDICATE_START_TOKEN)
        new_tokens.append(hacked_tokens[predicate_index])
        new_tokens.append(T2TQasemArgumentParser._PREDICATE_END_TOKEN)
        # put the rest of the sentence
        new_tokens.extend(hacked_tokens[(predicate_index + 1):])
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
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        post_processed = [
            self._postprocess(dec, item.sentence)
            for dec, item in zip(decoded, batch)
        ]
        return post_processed

    def _postprocess(self, decoded: str, tokens: TokenizedSentence):
        """
        Processes the generated output by a Text-to-Text model
        and parses it into a set of semantic arguments and their questions.
        :param decoded: The decoded output string produced by beam-search or other decoding algorithm.
        The correct format of the output is:
        <question_1>?answer_1;answer_2;<question_2>?<answer_1>...
        TODO(plroit): This format is ambiguous and hard to parse, need to re-train the model with a better format.
        :param tokens: The original, pre-tokenized, sentence tokens
        :return:
        """
        arguments = []

        hacked_tokens = T2TQasemArgumentParser.hack_unknown_tokens(tokens)

        # format: Who said something? they;him;What was said? some things
        splits = decoded.strip().split(";")
        curr_question = None
        for split in splits:
            if "?" in split:
                # this is a new question
                question_mark_idx = split.index("?")
                curr_question = split[:(question_mark_idx + 1)].strip()
                answer = split[(question_mark_idx + 1):].strip()
            else:
                # This is a new answer to an existing question
                answer = split.strip()
            if curr_question is None:
                # Must have at least one question
                continue
            # try to locate the answer in the original text
            answer_start, answer_end = find_answer_idx(hacked_tokens, answer)
            if answer_start is None:
                # try case insensitive
                answer_start, answer_end = find_answer_idx(
                    [tok.lower() for tok in hacked_tokens],
                    answer.lower())

            if answer_start is None:
                # ignore errors for now
                # errors.append({"q": curr_question, "a": answer, "text": " ".join(tokens)})
                continue
            arg_text = " ".join(tokens[answer_start: answer_end]),
            arg = QasemArgument(arg_text, curr_question, answer_start, answer_end)
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