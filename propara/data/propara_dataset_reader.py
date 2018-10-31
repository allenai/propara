from allennlp.common import tqdm
tqdm._tqdm.monitor_interval = 0

from typing import Dict, List, Tuple, Iterator
import collections
import csv
import enum
import itertools

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.instance import Instance
from allennlp.data.fields.field import Field
from allennlp.data.fields import TextField, IndexField, LabelField, SequenceField, ListField, SequenceLabelField, MetadataField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
_LEMMATIZER = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)


class Action(enum.Enum):
    NONE = 0
    CREATE = 1
    DESTROY = 2
    MOVE = 3

DOES_NOT_EXIST = '-'
UNKNOWN = '?'


def _infer_actions(states: List[str]) -> List[Action]:
    actions: List[Action] = []
    for prev_state, next_state in zip(states, states[1:]):
        if prev_state != DOES_NOT_EXIST and next_state == DOES_NOT_EXIST:
            # Existence -> Non-Existence
            action = Action.DESTROY
        elif prev_state == DOES_NOT_EXIST and next_state != DOES_NOT_EXIST:
            # Non-Existence -> Existence
            action = Action.CREATE
        elif prev_state != DOES_NOT_EXIST and next_state != prev_state:
            # Existence -> different Existence
            action = Action.MOVE
        else:
            # otherwise, no idea
            action = Action.NONE

        actions.append(action)

    return actions

Span = Tuple[int, int]
BeforeAfter = Tuple[Span, Span]
Tokenized = List[Token]


def _find_span(target: Tokenized, sentence: Tokenized, start: int=0, target_is_noun: bool=True) -> Span:
    """
    Returns the first span corresponding to `target` in `sentence`.
    Span indexes are inclusive.
    """
    if target_is_noun:
        target_tokens = [_LEMMATIZER(token.text.lower(), 'NOUN')[0] for token in target]
    else:
        target_tokens = [_LEMMATIZER(token.text.lower(), token.pos_)[0] for token in target]
    sentence_tokens = [_LEMMATIZER(token.text.lower(), token.pos_)[0] for token in sentence]

    target_length = len(target_tokens)

    if target_tokens == ['-']:
        # absent
        return (-3, -3)
    elif target_tokens == ['?']:
        # unknown
        return (-2, -2)

    for i in range(start, len(sentence_tokens)):
        if sentence_tokens[i:(i+target_length)] == target_tokens:
            return (i, i + target_length - 1)

    # not in sentence
    return (-1, -1)


def _compute_location_spans(states: List[Tokenized], sentences: List[Tokenized]) -> List[BeforeAfter]:
    results: List[BeforeAfter] = []

    for i, sentence in enumerate(sentences):
        before_loc = states[i]
        after_loc = states[i+1]

        before_span = _find_span(before_loc, sentence)
        after_span = _find_span(after_loc, sentence)

        results.append((before_span, after_span))

    return results


@DatasetReader.register("ProParaDatasetReader")
class ProParaDatasetReader(DatasetReader):
    """
    Reads a file from ProPara dataset. Each instance contains a paragraph, sentences and a list of participants.
    Labels consists of actions and state-values per participant per step.
    Input File Format: Example paragraph below:
        14	SID	PARTICIPANTS	water	water vapor	droplets	rain	snow
        14		PROMPT: What happens during the water cycle?	-=====	-=====	-=====	-=====	-=====
        14	state1		ocean , lake , river , swamp , and plant	-	-	-	-
        14	event1	Water from oceans, lakes, rivers, swamps, and plants turns into water vapor.
        14	state2		-	cloud	-	-	-
        14	event2	Water vapor forms droplets in clouds.
        14	state3		-	-	cloud	-	-
        14	event3	Water droplets in clouds become rain or snow and fall.
        14	state4		-	-	-	ground	ground
        14	event4	Some water goes into the ground.
        14	state5		ground	-	-	-	-
        14	event5	Some water flows down streams into rivers and oceans.
        14	state6		river and ocean	-	-	-	-

    Parameters
    ------------
    multiple_annotations: bool, optional (default = False)
        Do we have multiple annotations for each instance?
    token_indexers: Dict[str, TokenIndexer], optional (default = None)
        If not specified, we'll just use a SingleIdTokenIndexer.
    """

    def __init__(self,
                 multiple_annotations: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._multiple_annotations = multiple_annotations
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}


    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            # Group by id (or by nothing if there's an empty row).
            for group_id, group in itertools.groupby(reader, lambda row: row and row[0]):
                # If it's an empty row between examples, skip it.
                if not group_id:
                    continue

                # Otherwise generate the instances from this group.
                for instance in self._instances_from_group(group, group_id, file_path):
                    yield instance

    def _instances_from_group(self, group, group_id, file_path) -> Iterator[Instance]:
        participants = next(group)[3:]  # first row contains participants
        prompt = next(group)[2]         # second row contains the prompt

        # states[i][j] is the state of the i-th participant at time j
        states = [[] for _ in participants]

        sentence_texts = []

        for i, row in enumerate(group):
            label = row[1]
            if label.startswith('state'):
                # states start in the 4th column
                for j, state in enumerate(row[3:]):
                    states[j].append(state)
            elif label.startswith('event'):
                sentence_texts.append(row[2])
            else:
                raise ValueError(f"unknown row type {label}")

        if self._multiple_annotations:
            # In this case, each "state" is actually the pipe-delimited verdicts
            # of multiple annotators. So we need to
            #
            # 1. split on the pipe to get the annotations
            # 2. broadcast states that are ["-"] (this is an experimenter override)
            # 3. throw out annotators with incomplete results
            # 4. generate one instance per annotator
            # 5. complete with a score reflecting how 'common' their annotations were
            split_states = [[multi_state.split("|")
                             for multi_state in participant_states]
                            for participant_states in states]

            num_participants = len(split_states)
            num_states = len(split_states[0])

            # Sometimes annotations will be missing,
            # so we compute the number of "full" annotations
            num_annotations = min(len(multi_state)
                                  for participant_states in split_states
                                  for multi_state in participant_states
                                  if multi_state != ['-'])

            # Often we have 2 * num_annotations annotations, an "after" and "before".
            # The "after" ones are preferable, so we take those.
            # (Most of the time they're the same anyway.)
            split_states = [[multi_state[:num_annotations]
                             for multi_state in participant_states]
                            for participant_states in split_states]

            # Expand single dashes
            for i in range(num_participants):
                for j in range(num_states):
                    if split_states[i][j] == ['-']:
                        split_states[i][j] = ['-' for _ in range(num_annotations)]

            # Count how many times each annotation appears for each (participant,state) combination.
            counts = [[collections.Counter(multi_state)
                       for multi_state in participant_states]
                      for participant_states in split_states]

            for i in range(num_annotations):
                # Grab the states just for this annotator.
                annotator_states = [[multi_state[i]
                                     for multi_state in participant_states]
                                    for participant_states in split_states]

                # Compute a score based on how common the chosen states were.
                scores = [counts[i][j][annotator_states[i][j]] / num_annotations
                          for i in range(num_participants)
                          for j in range(num_states)]

                score = sum(scores) / len(scores)

                instance = self.text_to_instance(para_id=group_id,
                                                 sentence_texts=sentence_texts,
                                                 participants=participants,
                                                 states=annotator_states,
                                                 filename=file_path,
                                                 score=score)

                yield instance



        else:
            # Single annotation, so just create one instance.
            instance = self.text_to_instance(para_id=group_id,
                                             sentence_texts=sentence_texts,
                                             participants=participants,
                                             states=states,
                                             filename=file_path)

            yield instance

    @overrides
    def text_to_instance(self,
                         para_id: str,
                         sentence_texts: List[str],
                         participants: List[str],
                         states: List[List[str]] = None, # states[i][j] is ith participant at time j
                         filename: str = '',
                         score: float = None
                         ) -> Instance:

        tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))

        paragraph = " ".join(sentence_texts)

        # Tokenize the sentences
        sentences = [
            tokenizer.tokenize(sentence_text)
            for sentence_text in sentence_texts
        ]

        # Find the verbs
        verb_indexes = [
            [1 if token.pos_ == "VERB" else 0 for token in sentence]
            for sentence in sentences
        ]

        if states is not None:
            # Actions is (num_participants, num_events)
            actions = [_infer_actions(states_i) for states_i in states]

            tokenized_states = [
                [tokenizer.tokenize(state_ij) for state_ij in states_i]
                for states_i in states
            ]

            location_spans = [
                _compute_location_spans(states_i, sentences)
                for states_i in tokenized_states
            ]

        # Create indicators for the participants.
        participant_tokens = [
            tokenizer.tokenize(participant)
            for participant in participants
        ]
        participant_indicators: List[List[List[int]]] = []

        for participant_i_tokens in participant_tokens:
            targets = [list(token_group)
                        for is_semicolon, token_group in itertools.groupby(participant_i_tokens,
                                                                            lambda t: t.text == ";")
                        if not is_semicolon]

            participant_i_indicators: List[List[int]] = []

            for sentence in sentences:
                sentence_indicator = [0 for _ in sentence]

                for target in targets:
                    start = 0
                    while True:
                        span_start, span_end = _find_span(target, sentence, start, target_is_noun=True)
                        if span_start >= 0:
                            for j in range(span_start, span_end + 1):
                                sentence_indicator[j] = 1
                            start = span_start + 1
                        else:
                            break

                participant_i_indicators.append(sentence_indicator)

            participant_indicators.append(participant_i_indicators)

        fields: Dict[str, Field] = {}
        fields["paragraph"] = TextField(tokenizer.tokenize(paragraph), self._token_indexers)
        fields["participants"] = ListField([
                TextField(tokenizer.tokenize(participant), self._token_indexers)
                for participant in participants
        ])

        # One per sentence
        fields["sentences"] = ListField([
                TextField(sentence, self._token_indexers)
                for sentence in sentences
        ])

        # One per sentence
        fields["verbs"] = ListField([
            SequenceLabelField(verb_indexes[i], fields["sentences"].field_list[i])
            for i in range(len(sentences))
        ])
        # And also at the paragraph level
        fields["paragraph_verbs"] = SequenceLabelField(
            [verb_indicator for verb_indexes_i in verb_indexes for verb_indicator in verb_indexes_i],
            fields["paragraph"]
        )

        if states is not None:
            # Outer ListField is one per participant
            fields["actions"] = ListField([
                # Inner ListField is one per sentence
                ListField([
                    # action is an Enum, so call .value to get an int
                    LabelField(action.value, skip_indexing=True)
                    for action in participant_actions
                ])
                for participant_actions in actions
            ])

            # Outer ListField is one per participant
            fields["before_locations"] = ListField([
                # Inner ListField is one per sentence
                ListField([
                    SpanField(start, end, fields["sentences"].field_list[i])
                    for i, ((start, end), _) in enumerate(participant_location_spans)
                ])
                for participant_location_spans in location_spans
            ])
            # Outer ListField is one per participant
            fields["after_locations"] = ListField([
                # Inner ListField is one per sentence
                ListField([
                    SpanField(start, end, fields["sentences"].field_list[i])
                    for i, (_, (start, end)) in enumerate(participant_location_spans)
                ])
                for participant_location_spans in location_spans
            ])

        # one per participant
        fields["participant_indicators"] = ListField([
            # one per sentence
            ListField([
                SequenceLabelField(sentence_indicator, fields["sentences"].field_list[i])
                for i, sentence_indicator in enumerate(participant_i_indicators)
            ])
            for participant_i_indicators in participant_indicators
        ])

        # and also at the paragraph level
        # one per participant
        fields["paragraph_participant_indicators"] = ListField([
            SequenceLabelField([indicator
                                for sentence_indicator in participant_i_indicators
                                for indicator in sentence_indicator],
                               fields["paragraph"])
            for participant_i_indicators in participant_indicators
        ])

        # Finally, we want to indicate before / inside / after for each sentence.
        paragraph_sentence_indicators: List[SequenceLabelField] = []
        for i in range(len(sentences)):
            before_length = sum(len(sentence) for sentence in sentences[:i])
            sentence_length = len(sentences[i])
            after_length = sum(len(sentence) for sentence in sentences[(i+1):])
            paragraph_sentence_indicators.append(
                SequenceLabelField([0] * before_length + [1] * sentence_length + [2] * after_length,
                                   fields["paragraph"])
            )

        fields["paragraph_sentence_indicators"] = ListField(paragraph_sentence_indicators)

        # These fields are passed on to the decoder trainer that internally uses it
        # to compute commonsense scores for predicted actions
        fields["para_id"] = MetadataField(para_id)
        fields["participant_strings"] = MetadataField(participants)

        fields["filename"] = MetadataField(filename)

        if score is not None:
            fields["score"] = MetadataField(score)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ProParaDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop("token_indexers", {}))
        multiple_annotations = params.pop_bool("multiple_annotations", False)

        return ProParaDatasetReader(token_indexers=token_indexers, multiple_annotations=multiple_annotations)


