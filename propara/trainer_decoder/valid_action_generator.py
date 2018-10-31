from typing import List
import itertools

# The interface.
import math
import torch
from overrides import overrides

from propara.data.propara_dataset_reader import Action

ABLATE_HARD_CONSTRAINTS = False

class ValidActionGenerator(object):
    def __init__(self, name="stepper name not filled in yet."):
        self.name = name
        # Generates valid actions per participant "create/move/destroy" according to some rules.
        # Takes into account the previous state(s) in the action history.
        # e.g, if x is destroyed then for that participant, destroy is not a valid action.

    # returns shape: y x np
    #  y label combinations ( size of y = number of participants).
    #  count of y could be different for different paragraphs.
    def generate(self, action_history: List[List[int]], metadata={}) -> (List[List[int]], {}):
        raise NotImplementedError("Please Implement this method")


# A placeholder that does not prune the action space.
# creating many possible combinations of labels out of (num_labels ^ num_participants) possibilities.
class DummyConstrainedStepper(ValidActionGenerator):
    def __init__(self, num_labels, num_participants=-1):
        super().__init__('Dummy action generator that gives all combinations')
        self.num_labels = num_labels
        self.num_participants = num_participants

    def set_num_participants(self, num_participants):
        self.num_participants = num_participants

    @overrides
    def generate(self, action_history: List[List[int]], metadata={}) -> (List[List[int]], {}):
        valid_actions_per_participant: List[List[int]] = []
        # Generates all l^p combinations of l labels and p participants.
        for p in list(itertools.product(*[range(self.num_labels) for _ in range(self.num_participants)])):
            valid_actions_per_participant.append(list(p))
        return (valid_actions_per_participant, {})


# A placeholder that prunes the action space by discarding nonsensical actions.
# generating those  combinations of labels out of (num_labels ^ num_participants) possibilities, which are
# sensible according to a set of global commonsense rules.
class CommonsenseBasedActionGenerator(ValidActionGenerator):
    def __init__(self, num_labels, num_participants=-1, num_steps=-1, rules_used=[True, True, True, True]):
        super().__init__('Commonsense rules based action generator')
        self.num_labels = num_labels
        self.num_participants = num_participants
        self.num_steps = num_steps
        self.disallow_more_than_2_create_destroy = rules_used[0]
        self.disallow_more_than_half_participants_change_in_a_step = rules_used[1]
        self.disallow_one_participant_change_in_more_than_half_steps = rules_used[2]
        self.disallow_change_until_mentioned = rules_used[3]
        self.fraction_participants_can_change = 0.5
        self.fraction_steps_with_change = 0.5
        self.min_num_members = 4

    def set_num_participants(self, num_participants):
        self.num_participants = num_participants

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

    def set_min_num_members(self, min_num_members):
        self.min_num_members = min_num_members

    def get_rules_used(self):
        return [1 if self.disallow_more_than_2_create_destroy else 0,
                1 if self.disallow_more_than_half_participants_change_in_a_step else 0,
                1 if self.disallow_one_participant_change_in_more_than_half_steps else 0,
                1 if self.disallow_change_until_mentioned else 0
                ]

    def set_rules_used(self, rules_used, rule_2_fraction_participants=0.5, rule_3_fraction_steps=0.5):
        self.disallow_more_than_2_create_destroy = rules_used[0]
        self.disallow_more_than_half_participants_change_in_a_step = rules_used[1]
        self.disallow_one_participant_change_in_more_than_half_steps = rules_used[2]
        self.disallow_change_until_mentioned = rules_used[3]
        self.fraction_participants_can_change = rule_2_fraction_participants
        self.fraction_steps_with_change = rule_3_fraction_steps

    @overrides
    def generate(self, action_history: List[List[int]], metadata={}) -> (List[List[int]], {}):
        # Constraint 1: State values of a participant should be consistent across 1 column of the grid
        # Constraint 2: Not more than 2 CREATE/DESTROY in the single column
        valid_labels_per_participant = []
        valid_actions_list = []

        # Constraint: a participant cannot change state unless it is already introduced in the paragraph
        # size: num_parti * num_sent * sent_len
        participant_indicators = metadata['participant_indicators'] if 'participant_indicators' in metadata else None

        for p in range(self.num_participants):
            valid_labels_p = self.get_valid_labels_for_a_participant(p, action_history, participant_indicators)
            valid_labels_per_participant.append(valid_labels_p)

        # TODO: Add Constraint 3 when we have location spans
        # Constraint 3: if entity X is destroyed at location L, and immediately another entity Y is created,
        # then Y is usually created at the same location L

        # Generates all permutations of labels.
        # Constraint: in any step max k participants can change state (k=#participants/2)
        if self.disallow_more_than_half_participants_change_in_a_step:
            valid_actions_list = []
            for action in (list(itertools.product(*valid_labels_per_participant))):
                num_participants_can_change_right_now = self.get_num_from_frac(self.num_participants, self.fraction_participants_can_change)
                if self.get_num_participants_changed(action) <= num_participants_can_change_right_now:
                    valid_actions_list.append(action)

            return (valid_actions_list, {"rule_used": "blah"})
        else :
            return (list(itertools.product(*valid_labels_per_participant)), {"rule_used": "blah"})


    def get_num_from_frac(self, n, frac) -> float:
        return round(n * (1.0 if n <= self.min_num_members else frac))

    @staticmethod
    def get_num_participants_changed(action: List[int]) -> int:
        num_participants_changed = 0

        for p in range(len(action)):
            if not action[p] == Action.NONE.value:
                num_participants_changed += 1
        return num_participants_changed

    def get_valid_labels_for_a_participant(self,
                                           participant_id: int,
                                           action_history: List[List[int]],
                                           participant_indicators: torch.IntTensor=None) -> List[int]:

        valid_labels_p = []
        participant_mentioned = True

        # Constraint: If participant not yet mentioned, it cannot have a state change
        if self.disallow_change_until_mentioned and participant_indicators is not None:
            participant_mentioned = False
            num_steps_till_now = len(action_history)
            if num_steps_till_now > 0:
                mentioned = torch.sum(participant_indicators[participant_id][0:num_steps_till_now+1, :], dim=-1)
                if (torch.sum(mentioned, dim=-1)).cpu().data[0] > 0:
                            participant_mentioned = True
            if num_steps_till_now > 0 and not participant_mentioned:
                return [Action.NONE.value]

        # Constraint: a participant can change state in at most T/2 steps
        if self.disallow_one_participant_change_in_more_than_half_steps:
            num_changes_for_a_participant = self.num_changes_for_a_participant(participant_id, action_history)
            num_stets_with_change_right_now = self.get_num_from_frac(self.num_steps, self.fraction_steps_with_change)

            if float(num_changes_for_a_participant) >= num_stets_with_change_right_now:
                return [Action.NONE.value]

        num_creations_for_a_participant = self.num_labels_for_a_participant(participant_id, action_history,
                                                                            Action.CREATE.value)
        num_destructions_for_a_participant = self.num_labels_for_a_participant(participant_id, action_history,
                                                                               Action.DESTROY.value)

        # Constraint: State values of a participant should be consistent across 1 column of the grid
        # A participant can always undergo CREATE and NONE
        valid_labels_p.append(Action.NONE.value)

        # Constraint: A participant cannot exist before being created
        # Action sequence C (non-D)*  C on a single participant is not allowed
        if ABLATE_HARD_CONSTRAINTS or not self.exists_in_most_recent_step(participant_id, action_history):
            # Constraint: Not more than 2 CREATE/DESTROY in the single column
            if self.disallow_more_than_2_create_destroy:
                if num_creations_for_a_participant < 2:
                    valid_labels_p.append(Action.CREATE.value)
            else:
                valid_labels_p.append(Action.CREATE.value)

        # Constraint: A participant can undergo DESTROY and MOVE only of it exists
        # Action sequence D (non-C)*  D/M on a single participant is not allowed

        if ABLATE_HARD_CONSTRAINTS or not self.is_participant_destroyed_recently(participant_id, action_history):
            # Constraint 2: Not more than 2 CREATE/DESTROY in the single column
            if self.disallow_more_than_2_create_destroy:
                if num_destructions_for_a_participant < 2:
                    valid_labels_p.append(Action.DESTROY.value)
            else:
                valid_labels_p.append(Action.DESTROY.value)
            valid_labels_p.append(Action.MOVE.value)

        return valid_labels_p

    @staticmethod
    def exists_in_most_recent_step(participant_id: int, action_history: List[List[int]]) -> bool:
        if len(action_history) == 0:
            return False

        for time_step in range(len(action_history)-1, -1, -1):
            if action_history[time_step][participant_id] == Action.DESTROY.value:
                return False
            elif action_history[time_step][participant_id] == Action.MOVE.value:
                return True
            elif action_history[time_step][participant_id] == Action.CREATE.value:
                return True
            # skip None's
        return False

    @staticmethod
    def is_participant_destroyed_recently(participant_id: int, action_history: List[List[int]]) -> bool:
        if len(action_history) == 0:
            return False

        destroyed_recently = False
        for time_step in range(len(action_history)):
            if action_history[time_step][participant_id] == Action.DESTROY.value:
                destroyed_recently = True
            if action_history[time_step][participant_id] == Action.CREATE.value:
                destroyed_recently = False
        return destroyed_recently

    @staticmethod
    def num_labels_for_a_participant(participant_id: int, action_history: List[List[int]], label: int) -> bool:
        if len(action_history) == 0:
            return 0

        count = 0
        for time_step in range(len(action_history)):
            if action_history[time_step][participant_id] == label:
                count += 1

        return count

    @staticmethod
    def num_changes_for_a_participant(participant_id: int, action_history: List[List[int]]) -> bool:
        if len(action_history) == 0:
            return 0

        count = 0
        for time_step in range(len(action_history)):
            if action_history[time_step][participant_id] != Action.NONE.value:
                count += 1

        return count
