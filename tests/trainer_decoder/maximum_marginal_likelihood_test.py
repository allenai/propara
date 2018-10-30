# pylint: disable=invalid-name,no-self-use,protected-access
import math
from typing import List, Set, Dict, Sequence
from collections import defaultdict

from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding import DecoderState, DecoderStep

from propara.trainer_decoder.maximum_marginal_likelihood import MaximumMarginalLikelihood
from propara.trainer_decoder.propara_decoder_state import ProParaDecoderState
from propara.trainer_decoder.propara_decoder_step import ProParaDecoderStep


class TestMaximumMarginalLikelihood(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.initial_state = SimpleDecoderState(batch_indices=[0, 1],
                                                action_history=[[], []],
                                                score=[Variable(torch.Tensor([0.0])), Variable(torch.Tensor([0.0]))],
                                                start_values=[[0, 0], [1, 1]])
        self.decoder_step = SimpleDecoderStep()
        self.targets = torch.autograd.Variable(torch.Tensor([
            [
                [[2, 2], [3, 3], [4, 4]],
                [[1, 1], [3, 3], [4, 4]],
                [[1, 1], [2, 2], [4, 4]],
            ], [
                [[3, 3], [4, 4], [0, 0]],
                [[2, 2], [3, 3], [4, 4]],
                [[0, 0], [0, 0], [0, 0]],
            ]
        ]))

        # Dim. of target and target_mask are [torch.FloatTensor of size 2x3x3x2]
        self.target_mask = torch.autograd.Variable(torch.Tensor(
            [
                [
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1], [1, 1], [1, 1]]
                ],
                [
                    [[1, 1], [1, 1], [0, 0]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[0, 0], [0, 0], [0, 0]]
                ]
            ]))

        self.supervision = (self.targets, self.target_mask)
        # High beam size ensures exhaustive search.
        self.trainer = MaximumMarginalLikelihood()

    def test_decode(self):
        decoded_info = self.trainer.decode(self.initial_state, self.decoder_step, self.supervision)

        # Our loss is the negative log sum of the scores from each target sequence.  The score for
        # each sequence in our simple transition system is just `-sequence_length`.
        instance0_loss = math.log(math.exp(-3) * 3)  # all three sequences have length 3
        instance1_loss = math.log(math.exp(-2) + math.exp(-3))  # one has length 2, one has length 3
        expected_loss = -(instance0_loss + instance1_loss) / 2
        assert_almost_equal(decoded_info['loss'].data.numpy(), expected_loss)

    def test_create_allowed_transitions(self):
        result = self.trainer._create_allowed_transitions(self.targets, self.target_mask)
        # There were two instances in this batch.
        assert len(result) == 2

        # The first instance had six valid action sequence prefixes.
        assert len(result[0]) == 6
        print(result[0])
        assert result[0][()] == {(1, 1), (2, 2)}
        assert result[0][((1, 1),)] == {(2, 2), (3, 3)}
        assert result[0][((1, 1), (2, 2))] == {(4, 4)}
        assert result[0][((1, 1), (3, 3))] == {(4, 4)}
        assert result[0][((2, 2),)] == {(3, 3)}
        assert result[0][((2, 2), (3, 3))] == {(4, 4)}

        # The second instance had four valid action sequence prefixes.
        assert len(result[1]) == 4
        assert result[1][()] == {(2, 2), (3, 3)}
        assert result[1][((2, 2),)] == {(3, 3)}
        assert result[1][((2, 2), (3, 3))] == {(4, 4)}
        assert result[1][((3, 3),)] == {(4, 4)}

    def test_get_allowed_actions(self):
        state = DecoderState([0, 1, 0], [[1], [0], []], [])
        allowed_transitions = [{(1,): {2}, (): {3}}, {(0,): {4, 5}}]
        allowed_actions = self.trainer._get_allowed_actions(state, allowed_transitions)
        assert allowed_actions == [{2}, {4, 5}, {3}]


class SimpleDecoderState(DecoderState['SimpleDecoderState']):
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[List[int]]],
                 score: List[torch.autograd.Variable],
                 start_values: List[List[int]] = None) -> None:
        super().__init__(batch_indices, action_history, score)
        self.start_values = start_values or [[0, 0] for _ in range(len(batch_indices))]

    def is_finished(self) -> bool:
        return self.action_history[0][-1] == [4, 4]

    @classmethod
    def combine_states(cls, states) -> 'SimpleDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in
                            state.action_history]
        scores = [score for state in states for score in state.score]
        start_values = [start_value for state in states for start_value in state.start_values]
        return SimpleDecoderState(batch_indices, action_histories, scores, start_values)

    def __repr__(self):
        return f"{self.action_history}"


class SimpleDecoderStep(DecoderStep[SimpleDecoderState]):
    def __init__(self,
                 valid_actions: Set[int] = None,
                 include_value_in_score: bool = False):
        # The default allowed actions are adding 1 or 2 to the last element.
        self._valid_actions = valid_actions or {(1, 1), (2, 2)}
        # If True, we will add a small multiple of the action take to the score, to encourage
        # getting higher numbers first (and to differentiate action sequences).
        self._include_value_in_score = include_value_in_score

    def take_step(self,
                  state: SimpleDecoderState,
                  max_actions: int = None,
                  allowed_actions: List[Set] = None) -> List[SimpleDecoderState]:
        indexed_next_states: Dict[int, List[SimpleDecoderState]] = defaultdict(list)
        if not allowed_actions:
            allowed_actions = [None] * len(state.batch_indices)
        for batch_index, action_history, score, start_value, actions in zip(state.batch_indices,
                                                                            state.action_history,
                                                                            state.score,
                                                                            state.start_values,
                                                                            allowed_actions):

            prev_action1, prev_action2 = action_history[-1] if action_history else start_value
            for action1, action2 in self._valid_actions:
                next_item = (int(prev_action1 + action1), int(prev_action2 + action2))
                if actions and next_item not in actions:
                    continue
                new_history = action_history + [list(next_item)]
                # For every action taken, we reduce the score by 1.
                new_score = score - 1
                if self._include_value_in_score:
                    new_score += 0.01 * sum(next_item)
                new_state = SimpleDecoderState([batch_index],
                                               [new_history],
                                               [new_score])
                indexed_next_states[batch_index].append(new_state)
        next_states: List[SimpleDecoderState] = []
        for batch_next_states in indexed_next_states.values():
            sorted_next_states = [(-state.score[0].data[0], state) for state in batch_next_states]
            sorted_next_states.sort(key=lambda x: x[0])
            if max_actions is not None:
                sorted_next_states = sorted_next_states[:max_actions]
            next_states.extend(state[1] for state in sorted_next_states)
        return next_states
