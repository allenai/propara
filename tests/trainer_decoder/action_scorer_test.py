import pytest
from allennlp.common.testing import AllenNlpTestCase
from numpy.testing import assert_equal, assert_approx_equal

from propara.commonsense.background_knowledge.kb0_nothing import KB0Nothing
from propara.commonsense.background_knowledge.kb3_lexical import KBLexical
from propara.trainer_decoder.action_scorer import KBBasedActionScorer, ActionScorerDummy
import torch


class TestActionScorer(AllenNlpTestCase):
    def test_kb_action_scorer(self):
        lexical_kb = KBLexical()
        always_one_kb = KB0Nothing()

        action_scorer_kbnothing = KBBasedActionScorer(kb=always_one_kb, kb_coefficient=0.5)
        # action_scorer_kbnone should be identical to s_kbnothing
        action_scorer_kbnone = KBBasedActionScorer(kb=None, kb_coefficient=0.5)
        action_scorer_kblexical = KBBasedActionScorer(kb=lexical_kb, kb_coefficient=0.5)
        action_scorer_kblexical_coeff0 = KBBasedActionScorer(kb=lexical_kb, kb_coefficient=0.0)
        # action_scorer_dummy does not update any score.
        action_scorer_dummy = ActionScorerDummy()

        metadata = {"process_id": 514,  # How do glaciers form and move
                    "participant_strings": ["water", "glacier"]}
        action_logits = torch.autograd.Variable(torch.Tensor([[0.1, 0.7], [0.8, 0.2]]))  # two participants two steps.
        valid_actions = [1, 1]  # water action=1 (created), glacier action=1 (created)

        # These scorers don't really use action_history so pass None.
        s_kbnothing = action_scorer_kbnothing.score_of(action_history=None,
                                                       valid_action=valid_actions,
                                                       model_score=action_logits
                                                       )
        s_kb_eq_none = action_scorer_kbnone.score_of(action_history=None,
                                                       valid_action=valid_actions,
                                                       model_score=action_logits
                                                       )
        s_kblexical = action_scorer_kblexical.score_of(action_history=None,
                                                       valid_action=valid_actions,
                                                       model_score=action_logits,
                                                       metadata=metadata
                                                       )
        s_kblexical_coeff0 = action_scorer_kblexical_coeff0.score_of(action_history=None,
                                                                     valid_action=valid_actions,
                                                                     model_score=action_logits
                                                                     )
        s_dummy_scorer = action_scorer_dummy.score_of(action_history=None,
                                                      valid_action=valid_actions,
                                                      model_score=action_logits
                                                      )

        expected_kbnothing = torch.Tensor([0.5527, 0.3027])
        expected_kb_eq_none = torch.Tensor([0.5527, 0.3027])
        expected_kblexical = torch.Tensor([0.1103,  0.2613])
        expected_kblexical_coeff0 = torch.Tensor([0.7000, 0.2000])
        expected_dummy_scorer = torch.Tensor([0.7000, 0.2000])

        print(f"expected_kbnothing: {expected_kbnothing}\n"
              f"expected_kb_eq_none: {expected_kb_eq_none}\n"
              f"expected_kblexical: {expected_kblexical}\n"
              f"expected_kblexical_coeff0: {expected_kblexical_coeff0}\n"
              f"expected_dummy_scorer: {expected_dummy_scorer}\n")

        print(f"s_kbnothing: {s_kbnothing}\n"
              f"s_kbnone: {s_kb_eq_none}\n"
              f"s_kblexical: {s_kblexical}\n"
              f"s_kblexical_coeff0: {s_kblexical_coeff0}\n"
              f"s_dummy_scorer: {s_dummy_scorer}\n")

        # Expected scores are an approximation to 4 decimal places.
        approx_signif = 4
        for participant_id in [0, 1]:  # unclear how to use tensor comparison with approx_signif, so hard-coding.
            assert_approx_equal(s_kbnothing[participant_id], expected_kbnothing[participant_id], significant=approx_signif)
            assert_approx_equal(s_kb_eq_none[participant_id], expected_kb_eq_none[participant_id], significant=approx_signif)
            assert_approx_equal(s_kblexical[participant_id], expected_kblexical[participant_id], significant=approx_signif)
            assert_approx_equal(s_kblexical_coeff0[participant_id], expected_kblexical_coeff0[participant_id], significant=approx_signif)
            assert_approx_equal(s_dummy_scorer[participant_id], expected_dummy_scorer[participant_id], significant=approx_signif)

