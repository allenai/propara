import heapq
import logging

import torch
from allennlp.nn.decoding import DecoderStep

from propara.trainer_decoder.propara_decoder_state import ProParaDecoderState

from collections import defaultdict
from typing import List, Set, Dict

from overrides import overrides

from propara.trainer_decoder.action_scorer import ActionScorer
from propara.trainer_decoder.valid_action_generator import ValidActionGenerator, DummyConstrainedStepper, \
    CommonsenseBasedActionGenerator

logger = logging.getLogger(__name__)


class ProParaDecoderStep(DecoderStep[ProParaDecoderState]):
    def __init__(self,
                 action_scorer: ActionScorer,
                 valid_action_generator: ValidActionGenerator,
                 ):
        super(ProParaDecoderStep, self).__init__()
        self._action_scorer = action_scorer
        self._valid_action_generator = valid_action_generator
        self._num_failures = 0

    def change_action_scorer(self, new_action_scorer):
        self._action_scorer = new_action_scorer

    def get_action_scorer(self):
        return self._action_scorer

    def change_valid_action_generator(self, new_valid_action_generator):
        self._valid_action_generator = new_valid_action_generator

    def get_valid_action_generator(self):
        return self._valid_action_generator

    def set_num_participants(self, num_participants):
        if isinstance(self._valid_action_generator, DummyConstrainedStepper) or \
                isinstance(self._valid_action_generator, CommonsenseBasedActionGenerator):
            self._valid_action_generator.set_num_participants(num_participants)

    def set_num_steps(self, num_steps):
        if isinstance(self._valid_action_generator, CommonsenseBasedActionGenerator):
            self._valid_action_generator.set_num_steps(num_steps)

    @overrides
    def take_step(self,
                  state: ProParaDecoderState,
                  max_actions: int = None,
                  allowed_actions: List[Set] = None) -> List[ProParaDecoderState]:
        """
        Parameters:

        state: ``ProParaDecoderState``
            The state (or group of states) from which to choose a next step.
        max_actions: int
            The beam size
        allowed_actions: ``List[Set]``, optional (default = None)
            Valid actions is dynamic per state, while allowed actions is defined up front
            (e.g, list of labels such as CREATE DESTROY) for the entire .
            The base class (DecoderStep) enforces the shape `batch x allowed_actions_for_any_participant`.
            This is limiting because we need to specify allowed actions per participant
            So we actually need: List[Set[Ubertuple]] instead of List[Set[int]].

        Returns:
        ``List[ProParaDecoderState]``
            the next states
        """
        # Batch_index -> possible next states with their score.
        # 'ProParaDecoderState' object does not support indexing,
        # so we convert to a dictionary before sorting by score.
        indexed_next_states: Dict[int, List[ProParaDecoderState]] = defaultdict(list)

        if not allowed_actions:
            allowed_actions = [None] * len(state.batch_indices)

        # Generate a new state based on valid actions.
        # state.start_values can be None, so do not loop over it.
        for batch_index, action_hist, score, allowed_action in zip(state.batch_indices,
                                                                   state.action_history,
                                                                   state.score,
                                                                   allowed_actions):
            # Create many new ProParaDecoder states based on valid actions from the curr state.
            # Do not group the new states into one ProParaDecoderState.
            new_states = self.possible_states_from(batch_index,
                                                   state,
                                                   score,
                                                   # logits corresponding to the next step.
                                                   state.logit_tensor[batch_index][len(action_hist)],
                                                   action_hist,
                                                   allowed_action,
                                                   max_actions
                                                   )
            # Prepare for sorting by grouping returned states by their batch index.
            for new_state in new_states:
                indexed_next_states[batch_index].append(new_state)

        # Now, sort the states by score.
        next_states: List[ProParaDecoderState] = ProParaDecoderStep. \
            sort_states_by_score(indexed_next_states, max_actions)

        return next_states

    # Generate a list of states based on valid steps that can be taken from the current state.
    # To create a state, append history and compute a score for the state (per participant).
    def possible_states_from(self,
                             batch_index,
                             state,
                             state_score,
                             model_score,
                             action_history,
                             allowed_action,
                             max_actions) -> List[ProParaDecoderState]:
        new_states: List[ProParaDecoderState] = []
        scores_of_valid_actions = []
        sum_scores_of_valid_actions = 0.

        # In this function, one ProParaDecoderState contains exactly one state.
        # valid actions contain one action per participant e.g., (0: None, 1:Create, 0:None).
        (valid_actions, valid_actions_debug_info) = self._valid_action_generator.generate(action_history, state.metadata)

        if allowed_action and list(allowed_action)[0] not in valid_actions:
            self._num_failures += 1
            # FIXME wrong score of the valid allowed action
            for action in allowed_action:
                valid_actions.append(action)

        softmax_input = []  # Array of 1D tensors to be stacked for log(softmax)
        for valid_action_id, valid_action in enumerate(valid_actions):
            # Compute score per participant.
            curr_scores_per_parti = (self._action_scorer.score_of(
                action_history,
                valid_action,
                model_score,
                state.metadata))

            # participant wise score.
            # It is unclear if we need to append the history of prev. participant wise scores.
            # because we are already maintaining a score per state.
            # scores_of_valid_actions.append(curr_scores_per_parti + previous_scores_per_participant)
            scores_of_valid_actions.append(curr_scores_per_parti)
            softmax_input.append(curr_scores_per_parti.sum())  # 1D Tensor
            sum_scores_of_valid_actions += curr_scores_per_parti.sum()

        curr_scores = torch.nn.functional.log_softmax(torch.cat(softmax_input), dim=-1)
        # allowed valid action (that is part of the gold sequence)
        for valid_action_id, valid_action in enumerate(valid_actions):
            in_beam_search = state.metadata.get('in_beam_search', False)
            valid_action_is_allowed = allowed_action and valid_action in allowed_action
            if in_beam_search or valid_action_is_allowed:
                # num participants should match.
                # logit_tensor: (batch_size, num_sentences, num_participants, num_actions)
                # assert len(new_scores) == state.logit_tensor.shape[2]

                # lookup per participant score of this chosen valid_action_id.
                unnorm_score_per_participant = scores_of_valid_actions[valid_action_id]
                curr_score = curr_scores[valid_action_id]

                # new state score = prev state score (must be maintained for backprop) + curr score
                new_state_score = state_score + curr_score

                # Note: We only construct new states for the chosen valid actions.
                # For these new states, the group size is 1 (so we construct [batch_index] etc.).
                new_state = ProParaDecoderState(
                    # Shape: [batch]
                    group_indices=[batch_index],
                    # Shape: [batch[step[participants_labels]]]
                    action_history=[action_history + [valid_action]],
                    # Shape: [batch[participant_label_score]]
                    participant_score_per_group=[unnorm_score_per_participant],
                    participants_embedding=state.participants_embedding,
                    logit_tensor=state.logit_tensor,
                    instance_id=state.instance_id,
                    metadata=state.metadata,
                    overall_score=[new_state_score]  # score of the state.
                    # start_value was only needed for step 1.
                )

                new_states.append((-new_state_score.data[0], new_state))
                new_states.sort(key=lambda pair: pair[0])
                new_states = new_states[:max_actions]

        return [pair[1] for pair in new_states]

    @classmethod
    def sort_states_by_score(
            cls,
            indexed_next_states,
            max_actions
    ) -> List[ProParaDecoderState]:
        next_states: List[ProParaDecoderState] = []
        # state is of type ProParaDecoderState -- state has group size of 1
        # sort these states based on state score (which is a list of autograd variables).

        for batch_next_states in indexed_next_states.values():
            sorted_next_states = [(-state.score[0].data[0], state) for state in batch_next_states]
            sorted_next_states.sort(key=lambda x: x[0])
            if max_actions is not None:
                sorted_next_states = sorted_next_states[:max_actions]
            next_states.extend(state[1] for state in sorted_next_states)
        return next_states
