import torch
from typing import List, Tuple
from overrides import overrides

from qanom.question_info import get_slots, get_role

from .common_defs import ArgInputExample, iter_batches, QasemFrame, QasemArgument, TokenizedSentence

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerBase

from .torch_utils import get_device


def _span_intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    max_start = max(a[0], b[0])
    min_end = min(a[1], b[1])
    inter_length = max(0, min_end - max_start)
    return inter_length


def _span_distance(span1: Tuple[int, int], span2: Tuple[int, int]):
    s1, e1 = span1
    s2, e2 = span2
    if _span_intersection((s1, e1), (s2, e2)) > 0:
        return 0
    # no overlap, if span_1 is first, take s2 - e1
    if s1 < s2:
        return s2 - e1
    else:
        return s1 - e2


def get_closest_span(spans: list[tuple[int, int]], index: int) -> tuple[int, int]:
    my_span = index, index+1
    distances = [_span_distance(span, my_span) for span in spans]
    min_idx = 0
    for idx, dist in enumerate(distances):
        if dist < distances[min_idx]:
            min_idx = idx
    return spans[min_idx]


def find_answer_idx_with_fallback(tokens: List[str], answer: str) -> list[tuple[int, int]]:
    # try to locate the answer in the original text
    found_spans = find_answer_idx(tokens, answer)
    if not found_spans:
        # try case insensitive
        found_spans = find_answer_idx(
            [tok.lower() for tok in tokens], answer.lower()
        )
    return found_spans


def find_answer_idx(tokens: List[str], answer: str) -> list[tuple[int, int]]:
    # If the text token is in the prefix of the answer
    possible_starts = [idx for idx, tok in enumerate(tokens)
                       if answer[:len(tok)] == tok]
    locations_found = set()
    last_end_idx = -1
    for first_token_idx in possible_starts:
        if first_token_idx < last_end_idx:
            continue
        end_token_idx = find_answer_from_token(tokens, first_token_idx, answer)
        if end_token_idx is not None:
            locations_found.add((first_token_idx, end_token_idx))
            # start next search after this ending token?
            last_end_idx = end_token_idx

    return sorted(locations_found)


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
    _ANSWER_SEPARATOR = "<extra_id_3>"

    _PARSE_PREFIX_TOKENS = ["Generate",  "QA",  "pairs:"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 batch_size=_DEFAULT_BATCH_SIZE,
                 num_beams=_DEFAULT_NUM_BEAMS,
                 max_length=_DEFAULT_MAX_LENGTH,
                 predicate_start_token=_PREDICATE_START_TOKEN,
                 predicate_end_token=_PREDICATE_END_TOKEN,
                 qa_separator=_QA_SEPARATOR,
                 answer_separator=_ANSWER_SEPARATOR,
    ):
        """

        :param model: The encoder-decoder model to use.
        :param tokenizer: An instance of the tokenizer to use
        :param batch_size: The number of examples to process concurrently in a batch.
        :param num_beams: The number of beams in beam-search to use in decoding.
        :param max_length: Maximum length of the generated output Q&A pairs.
        :param predicate_start_token: a marker token to designate the predicate
        :param predicate_end_token: a marker token to designate the predicate
        :param qa_separator: a marker token to distinguish different QA pairs
        :param ansewr_separator: a marker token to distinguish different answers within a QA pair.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length
        self.predicate_start_end_markers = predicate_start_token, predicate_end_token
        self.qa_separator = qa_separator
        self.answer_separator = answer_separator

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, path_or_model_name: str, device: str = None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path_or_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_or_model_name)
        device = get_device(device=device)
        model = model.to(device)
        return cls(model, tokenizer, **kwargs)
    
    def _prepare_prompt(self, sample: ArgInputExample):
        # prompt:
        # "Generate QA pairs: The fox <extra_id_0> jumped <extra_id_0> over the fence"
        tokens = sample.sentence
        predicate_index = sample.predicate.index

        # Prefix that starts the prompt (our T5-model was trained with this prefix)
        new_tokens = T2TQasemArgumentParser._PARSE_PREFIX_TOKENS[:]
        # take the sentence till the predicate token
        new_tokens.extend(tokens[:predicate_index])
        # put the predicate token between two special tokens marking start and end.
        # The model is trained without spaces between the special tokens and the predicate
        pred_start_marker, pred_end_marker = self.predicate_start_end_markers
        marked_predicate = "".join([pred_start_marker,
                                    tokens[predicate_index],
                                    pred_end_marker])
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
            self._postprocess(dec, item)
            for dec, item in zip(decoded, batch)
        ]
        return post_processed

    def _postprocess(self, decoded: str, sample: ArgInputExample) -> list[QasemArgument]:
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
        tokens = sample.sentence
        qa_pairs = decoded2.split(self.qa_separator)
        for raw_qa_pair in qa_pairs:
            qa_splits = raw_qa_pair.split("?", maxsplit=1)
            if len(qa_splits) <= 1:
                continue
            raw_question = qa_splits[0].strip() + "?"
            role = get_role(raw_question)
            if role:
                # let's not use qanom.SemanticRole enum
                # it is coupled with prepositions in a specific dataset
                # instead of representing the role as a syntactic position
                # such as R0, R1, R2 or an adjunct and an optional preposition
                role = role.name
            # question, role = self._parse_question(question)
            answers = qa_splits[1].split(self.answer_separator)
            answers = [ans.strip() for ans in answers]
            for answer in answers:
                # try to locate the answer in the original text
                found_locations = find_answer_idx_with_fallback(tokens, answer)
                if not found_locations:
                    continue
                answer_start, answer_end = get_closest_span(found_locations, sample.predicate.index)
                arg_text = " ".join(tokens[answer_start: answer_end])
                arg = QasemArgument(arg_text, raw_question, answer_start, answer_end, role)
                arguments.append(arg)

        return arguments

    def predict(self, items: List[ArgInputExample]) -> List[QasemFrame]:
        if not items:
            return []
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
    

class T2TPropBankArgumentParser(T2TQasemArgumentParser):

    _CORE = [
        "A0", "A1", "A2", "A3", "A4", "A5", "AA",
    ]

    _MODIFIERS = [
        "ADJ", "ADV", "CAU", "COM", "DIR", "DIS", "DSP",
        "EXT", "GOL", "LVB", "LOC", "MNR", "MOD", "NEG", 
        "PNC", "PRD", "PRP", "PRR", "PRX", "TMP", "REC"
    ]
    _KNOWN_ROLES = [
        *_CORE,
        *[f"R-{r}" for r in _CORE],
        *[f"C-{r}" for r in _CORE],
        *[f"AM-{r}" for r in _MODIFIERS],
        *[f"R-AM-{r}" for r in _MODIFIERS],
        *[f"C-AM-{r}" for r in _MODIFIERS],
    ]

    def _split_role_and_args(self, raw_role_and_args):
        raw_role_and_args = raw_role_and_args.strip()
        role = None
        raw_args = ""
        for known_role in self._KNOWN_ROLES:
            if raw_role_and_args.startswith(known_role):
                role = known_role
                # expected format: A0 : my_argument here
                raw_args: str = raw_role_and_args[len(known_role):]
                raw_args = raw_args.strip()
                if raw_args.startswith(":"):
                    raw_args = raw_args.replace(":", "", 1).strip()
                break
        # Encountered an unknown role. Keep argument nevertheless.
        if role is None:
            role = "<UNK>"
        
        role_args = [arg.strip() for arg in raw_args.split(self.answer_separator)]
        return role, role_args

    @overrides
    def _postprocess(self, decoded: str, sample: ArgInputExample) -> list[QasemArgument]:
        arguments = []
        # we did not skip special tokens because we rely on <extra_id_2>
        # now need to remove other special tokens by ourselves
        decoded2 = decoded.replace(
            self.tokenizer.pad_token, "").replace(
            self.tokenizer.eos_token, "").strip(
        )
        tokens = sample.sentence
        roles_and_args = decoded2.split(self.qa_separator)
        for raw_role_and_args in roles_and_args:
            # now, we forgot that during training..
            role, raw_args = self._split_role_and_args(raw_role_and_args)
            for raw_arg in raw_args:
                # try to locate the answer in the original text
                answer_spans = find_answer_idx_with_fallback(tokens, raw_arg)
                answer_start, answer_end = get_closest_span(answer_spans, sample.predicate.index)
                if answer_start is None:
                    continue
                arg_text = " ".join(tokens[answer_start: answer_end])
                arg = QasemArgument(arg_text, "", answer_start, answer_end, role)
                arguments.append(arg)

        return arguments