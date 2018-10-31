from typing import List

# The interface.
import math
import torch
import torch.nn.functional as f
from overrides import overrides

from propara.commonsense.background_knowledge.kb import KB
from propara.commonsense.background_knowledge.kb0_nothing import KB0Nothing
from propara.data.propara_dataset_reader import Action


class ActionScorer(object):
    def __init__(self, name="state scorer name not filled in yet."):
        self.name = name

    def score_of(self, action_history, valid_action, model_score, metadata={}) -> List[float]:
        """
        Updates model score.
        action_history: List[List[int]] [steps[participants]]
        valid_action: shape # (num_participants,)
        model_score: shape # (num_participants, num_actions)
        metadata: a dict with any scorer specific info. e.g. commonsense scorer requires topic name
        and an initial version of it requires participant names.
        Returns: for every participant returns one score.
        """

        raise NotImplementedError("Please Implement this method")


class ActionScorerDummy(ActionScorer):
    """
    A placeholder that always returns a score of 1.0 * logit_score.
    """
    def __init__(self):
        super().__init__('Dummy scorer')

    @overrides
    def score_of(self,
                 action_history,  # steps, participants
                 valid_action,  # (num_participants,)
                 model_score,  # (num_participants, num_actions)
                 metadata={}) -> torch.autograd.Variable:
        if model_score is not None:
            # model_score shape: num_participants * num_actions
            # Suppose num_parti = 2, num_actions = 2
            # Model score: [torch.FloatTensor of size 2x2]
            #   0.1000 0.7000
            #   0.8000 0.2000
            # Valid actions: [0, 1]
            # Mask: [torch.FloatTensor of size 2x2]
            #   1  0
            #   0  1
            # The mask tells which valid actions are selected,
            # so we can return scores of participants [0.1, 0.2] in this case.
            mask = model_score.data.new(model_score.size()).fill_(0)
            for participant_idx, action in enumerate(valid_action):
                mask[participant_idx][action] = 1
            # This dummy scorer does not update the score.
            # just returns a vector of scores  (i.e., 0.1, 0.2) in this case.
            #   0.1  0
            #   0    0.2
            new_score = (model_score * torch.autograd.Variable(mask)).sum(dim=-1)

            return new_score
        else:
            raise RuntimeError("Action scorer requires that model scores cannot be empty.")

class KBBasedActionScorer(ActionScorer):
    """
    commonsense based action scorer
    """
    def __init__(self, kb: KB, kb_coefficient):
        self.kb = kb  # KBLexical, KB0Nothing, any new KB that can provide prob_of().
        self.kb_coefficient = kb_coefficient
        super().__init__(f"commonsense based action scorer ({self.kb.name if kb else 'No KB provided.'})")
        # Simple temporary checks to verify that commonsense is loaded correctly.
        # Move these to test cases instead.
        # assert self.kb.kb_partials[2019]['create'][4][0] == 'web'
        # assert self.kb.prob_of(514, ('glacier', 'create', '-', '?', 0)) > 0.1

    @overrides
    def score_of(self, action_history, valid_action, model_score, metadata={}) -> List[float]:
        if model_score is None:
            raise RuntimeError("Action scorer requires that model scores cannot be empty.")

        process_id = int(metadata["process_id"]) \
            if metadata is not None and "process_id" in metadata else -1
        participant_strings = metadata["participant_strings"] \
            if metadata is not None and "participant_strings" in metadata else []

        # From model_score
        #   0.1  0.7
        #   0.8  0.2
        # based on valid action [0, 1]
        #  i.e. participant p 1 valid action a = 0; p2 a=1
        mask = model_score.data.new(model_score.size()).fill_(0)
        for participant_idx, action in enumerate(valid_action):
            mask[participant_idx][action] = 1
        # After applying mask, convert the new score to a vector.
        #   0.1  0
        #   0    0.2
        # that gets converted to [0.1, 0.2]
        new_score = (model_score * torch.autograd.Variable(mask)).sum(dim=-1)

        # model_score shape: num_participants * num_actions
        # (already sliced for the current step id).
        score_per_participant = [(parti_id, label_id, new_score[parti_id])
                                 for parti_id, label_id in enumerate(valid_action)]

        kb_boosts = torch.autograd.Variable(
            model_score.data.new(model_score.size()[0]).fill_(1))
        for parti_id, label_id, score in score_per_participant:
            # lookup_tuple example: (snow, MOVE, ?, area, 0.691)
            # Temporarily, get the participant strings,
            # eventually, take embeddings instead of participant names and
            # take the nearest neighbor in the kb for that process topic.
            participant = (participant_strings[parti_id]
                           if participant_strings and parti_id < len(participant_strings)
                           else "")
            try:
                lookup_tuple = (participant, str(Action(label_id).name), '', '', score)
                kb_boost = (self.kb.prob_of(process_id, lookup_tuple)
                            if participant and label_id >= 0 and process_id >= 0
                            else 1.0)  # 1.0 is backoff.
                kb_boosts[parti_id] = 1.0 if math.isnan(kb_boost) or kb_boost < 0.0 else kb_boost
            except (AttributeError, TypeError, IndexError, KeyError) as exc:
                print("Exception: ", exc, "\n\tProbably commonsense lookup failed for process_id:",
                      process_id, " label_id", label_id, " participant id:", parti_id,
                      'participant_strings', participant_strings)
                # If commonsense look up fails do not penalize.
                kb_boosts[parti_id] = 1.0

        # kb_boost = 1.0 (very good prediction so reward it)
        #            0.41 is the max boost
        # kb_boost = 0.5 (not very good prediction)
        #            -0.7 is the min boost
        # kb_coefficient * kb_score + (1-kb_coefficient) * model_score

        final_value_1 = torch.mul(new_score, (1 - self.kb_coefficient))
        final_value_2_ii = torch.log(0.5 + kb_boosts)
        final_value_2 = torch.mul(final_value_2_ii, self.kb_coefficient)
        final_value = final_value_1.add(final_value_2)
        return final_value
