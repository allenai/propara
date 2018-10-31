# pylint: disable=invalid-name,no-self-use,protected-access
import numpy
import torch
import pytest

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding import BeamSearch
from torch.autograd import Variable

from propara.trainer_decoder.action_scorer import ActionScorerDummy, KBBasedActionScorer
from propara.trainer_decoder.propara_decoder_state import ProParaDecoderState
from propara.trainer_decoder.propara_decoder_step import ProParaDecoderStep
from propara.trainer_decoder.valid_action_generator import DummyConstrainedStepper


class TestProParaDecoderStep(AllenNlpTestCase):

    # @pytest.mark.skip()
    def test_with_model_scores_2steps(self):
        beam_search = BeamSearch.from_params(Params({'beam_size': 2}))
        max_num_steps = 2
        num_participants = 2
        num_labels = 2  # (Create/0, Destroy/1)

        # supplies parti embedding: group_size, nparti, num_tokens_p * token_embedding_dim
        # 1, num_p = 2, max_tokens_p = 1, token_embedding= size 3
        participants_embedding = [[[0.11, 0.21, 0.31]], [[0.12, 0.22, 0.32]]]
        # shape: 2 (batchsize), 2 (steps), 3 (participants), 4 (labels)
        #
        action_logits = [
            [[0.1, 0.7], [0.8, 0.2]],
            [[0.3, 0.4], [0.4, 0.6]]
        ]

        logit_tensor = torch.autograd.Variable(torch.Tensor([action_logits]))

        grouped_initial_state = ProParaDecoderState(
            # 2 labels (C, D) ^2 participants=4 states per step.
            # These four can be grouped into one grouped_initial_state
            group_indices=[0],
            # Initially, there is no action history per group item.
            action_history=[
                [],  # history for group 0
                # []  # history for group 1
            ],
            # shape (num_groups, num_participants)
            participant_score_per_group=[torch.autograd.Variable(torch.Tensor([0.0 for _ in range(num_participants)]))],
            # # Initial scores for all "num_participants" participants.
            # # Perhaps we should read these off from model_scores.
            # # or call action_scorer on the initial state?
            # participant_score_per_group=[
            #     [],  # score for group 0
            #     # []  # score for group 1
            # ],
            participants_embedding=participants_embedding,
            # action_logits=action_logits,
            # shape: 2 (batchsize), 2 (steps), 3 (participants), 4 (labels)
            logit_tensor=logit_tensor,
            instance_id=None,
            metadata={'in_beam_search': True},
            # Initial states: containing labels for every participant.
            # Perhaps the action generator should generate all possible states for step 0.
            # and start_values be None.
            overall_score=None,
            start_values=None
        )

        decoder_step = ProParaDecoderStep(
            # KBBasedActionScorer(),
            ActionScorerDummy(),
            DummyConstrainedStepper(num_labels, num_participants)
        )

        best_states = beam_search.search(
            max_num_steps,
            grouped_initial_state,
            decoder_step,
            keep_final_unfinished_states=False
        )

        # "best state" for all instances batch_wise
        # This passed before adding action logits.
        for k, v in best_states.items():
            # v[0] is the highest scored state in this step.
            # action_history[0] is the zeroth group element (note that the finished states have exactly one group elem)
            if k == 0:
                assert v[0].action_history[0] == [[1, 0], [1, 1]]
                assert v[0].score[0].data[0] == pytest.approx(-2.117511510848999)
        assert len(best_states) == 1

    def test_with_model_scores_3steps(self):
        beam_search = BeamSearch.from_params(Params({'beam_size': 1}))
        max_num_steps = 3
        num_participants = 2
        num_labels = 2  # (Create/0, Destroy/1)

        # supplies parti embedding: group_size, nparti, num_tokens_p * token_embedding_dim
        # 1, num_p = 2, max_tokens_p = 1, token_embedding= size 3
        participants_embedding = [[[0.11, 0.21, 0.31]], [[0.12, 0.22, 0.32]]]
        # shape: 2 (batchsize), 2 (steps), 3 (participants), 4 (labels)
        #
        action_logits = [
            [[0.1, 0.7], [0.8, 0.2]],
            [[0.3, 0.4], [0.4, 0.6]],
            [[0.2, 0.4], [0.5, 0.4]]
        ]

        logit_tensor = torch.autograd.Variable(torch.Tensor([action_logits]))

        grouped_initial_state = ProParaDecoderState(
            # 2 labels (C, D) ^2 participants=4 states per step.
            # These four can be grouped into one grouped_initial_state
            group_indices=[0],
            # Initially, there is no action history per group item.
            action_history=[
                [],  # history for group 0
                # []  # history for group 1
            ],
            # shape (num_groups, num_participants)
            participant_score_per_group=[torch.autograd.Variable(torch.Tensor([0.0 for _ in range(num_participants)]))],
            # # Initial scores for all "num_participants" participants.
            # # Perhaps we should read these off from model_scores.
            # # or call action_scorer on the initial state?
            # participant_score_per_group=[
            #     [],  # score for group 0
            #     # []  # score for group 1
            # ],
            participants_embedding=participants_embedding,
            # action_logits=action_logits,
            # shape: 2 (batchsize), 2 (steps), 3 (participants), 4 (labels)
            logit_tensor=logit_tensor,
            instance_id=None,
            metadata={'in_beam_search': True},
            # Initial states: containing labels for every participant.
            # Perhaps the action generator should generate all possible states for step 0.
            # and start_values be None.
            overall_score=None,
            start_values=None
        )

        decoder_step = ProParaDecoderStep(
            # KBBasedActionScorer(),
            ActionScorerDummy(),
            DummyConstrainedStepper(num_labels, num_participants)
        )

        best_states = beam_search.search(
            max_num_steps,
            grouped_initial_state,
            decoder_step,
            keep_final_unfinished_states=False
        )

        # "best state" for all instances batch_wise
        # This passed before adding action logits.
        for k, v in best_states.items():
            # v[0] is the highest scored state in this step.
            # action_history[0] is the zeroth group element (note that the finished states have exactly one group elem)
            if k == 0:
                assert v[0].action_history[0] == [[1, 0], [1, 1], [1, 0]]
                #assert v[0].score[0].data[0] == pytest.approx(-2.117511510848999)
        assert len(best_states) == 1

