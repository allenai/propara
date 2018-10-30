# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import AllenNlpTestCase

from propara.data.propara_dataset_reader import Action
from propara.trainer_decoder.valid_action_generator import DummyConstrainedStepper, CommonsenseBasedActionGenerator


class TestValidActionGenerator(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.action_history_step0 = []
        self.action_history_step1 = [[Action.NONE.value, Action.DESTROY.value]]
        self.action_history_step2 = [[Action.NONE.value, Action.DESTROY.value],
                                     [Action.DESTROY.value, Action.NONE.value]]
        self.action_history_step4 = [[Action.NONE.value, Action.DESTROY.value],
                                     [Action.DESTROY.value, Action.CREATE.value],
                                     [Action.CREATE.value, Action.DESTROY.value],
                                     [Action.DESTROY.value, Action.CREATE.value]]
        self.dummy_constrained_stepper = DummyConstrainedStepper(len(Action), 2)
        self.commonsense_action_generator = CommonsenseBasedActionGenerator(len(Action),
                                                                            num_participants= 2,
                                                                            num_steps= 4,
                                                                            rules_used=
                                                                             [True,  # C/D/C/D cannot happen
                                                                             True,  # > 1/2 partic
                                                                             False,  # > 1/2 steps cannot change
                                                                             False  # until mentioned
                                                                             ]
                                                                           )
        self.commonsense_action_generator.set_min_num_members(1)

    def test_dummy_constrained_stepper(self):
        (valid_actions_at_step1, valid_actions_debug_info) = self.dummy_constrained_stepper.generate(action_history=self.action_history_step1)
        assert valid_actions_at_step1 == [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0],
                                          [1, 1], [1, 2], [1, 3], [2, 0], [2, 1],
                                          [2, 2], [2, 3], [3, 0], [3, 1], [3, 2],
                                          [3, 3]]

    def test_common_sense_action_generator_rules_used(self):
        self.commonsense_action_generator.set_rules_used([False, # C/D/C/D cannot happen
                                                         False, # > 1/2 partic
                                                         False, # > 1/2 steps cannot change
                                                         False  # until mentioned
                                                         ])
        (valid_actions_at_step0, valid_actions_debug_info_0) = self.commonsense_action_generator.generate(action_history=self.action_history_step0)
        print("valid_actions_at_step0: ", valid_actions_at_step0)
        assert valid_actions_at_step0 == \
               [(0, 0), (0, 1), (0, 2), (0, 3),
                (1, 0), (1, 1), (1, 2), (1, 3),
                (2, 0), (2, 1), (2, 2), (2, 3),
                (3, 0), (3, 1), (3, 2), (3, 3)]

        self.commonsense_action_generator.set_rules_used([False,  # C/D/C/D cannot happen
                                                          True,  # > 1/2 partic
                                                          False,  # > 1/2 steps cannot change
                                                          False  # until mentioned
                                                          ])
        (valid_actions_at_step1, valid_actions_debug_info_1) = self.commonsense_action_generator.generate(
            action_history=self.action_history_step1)
        print("valid_actions_at_step1: ", valid_actions_at_step1)
        # P2 is destroyed so it cannot be moved, P1 can undergo all actions, both P1 and P2 cannot change at the same time
        assert valid_actions_at_step1 == [(0, 0), (0, 1),
                                          (1, 0),
                                          (2, 0),
                                          (3, 0)]

        self.commonsense_action_generator.set_rules_used([False,  # C/D/C/D cannot happen
                                                          True,  # > 1/2 partic
                                                          False,  # > 1/2 steps cannot change
                                                          False  # until mentioned
                                                          ])
        (valid_actions_at_step4, valid_actions_debug_info_4) = self.commonsense_action_generator.generate(
            action_history=self.action_history_step4)
        print("valid_actions_at_step4: ", valid_actions_at_step4)
        # P1 does not exists it can only be created and none
        # P2 can be D/M/N (because P2 is recently created)
        # both cannot change at the same time
        assert valid_actions_at_step4 == [(0, 0), (0, 2), (0, 3),
                                          (1, 0)]

        self.commonsense_action_generator.set_rules_used([True,  # C/D/C/D cannot happen
                                                          True,  # > 1/2 partic
                                                          False,  # > 1/2 steps cannot change
                                                          False  # until mentioned
                                                          ])
        (valid_actions_at_step4, valid_actions_debug_info_4) = self.commonsense_action_generator.generate(
            action_history=self.action_history_step4)
        print("valid_actions_at_step4: ", valid_actions_at_step4)
        # P1 does not exists, destroyed twice, P1 can only be created, and none
        # P2 exists, created twice, P2 can only be none, or move
        # P1, P2 dont change at the same time
        assert valid_actions_at_step4 == [(0, 0), (0, 3), (1, 0)]

        self.commonsense_action_generator.set_rules_used([True,  # C/D/C/D cannot happen
                                                          True,  # > 1/2 partic
                                                          True,  # > 1/2 steps cannot change
                                                          False  # until mentioned
                                                          ])
        (valid_actions_at_step4, valid_actions_debug_info_4) = self.commonsense_action_generator.generate(
            action_history=self.action_history_step4)
        print("valid_actions_at_step4: ", valid_actions_at_step4)
        # P1, P2 have both changed at least twice, so there is no change possible
        assert valid_actions_at_step4 == [(0, 0)]

