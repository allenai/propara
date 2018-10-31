from allennlp.nn.decoding import DecoderState
from typing import List
import torch


# This data structure contains grouped states to be used in beam search.
# Suppose group size = 1, then, ProParaDecoderState consists of the per
# participant label/score, and a history of steps taken so far.
# To align with the Beam search code, we sum the scores of all participants.
# This score is weighted using whatever ActionScorer the DecoderStep uses.
#
####################
# An example:
####################
#   3 sentences or time-steps (s1, s2, s3),
#   2 participants (p1, p2),
#   3 possible labels for each participant (0=CREATE, 1=DESTROY, 2=NONE)
#   We assume the following computations for one state in a group of states.
#
####################
# Logits from model:
####################
# For simplification, actual scores instead of log values are considered here.
#         p1                       p2
# s1     [0:0.8, 1:0.6, 2:0.1]    [0:0.4, 1:0.9, 2:0.1]
# s2     [0:0.3, 1:0.2, 2:0.8]    [0:0.1, 1:0.2, 2:0.9]
#
#######################
# Converted to 9 states:
#######################
#       step1            step2                        step 3
#       [0, 0]:0.32      [0, 0]:0.32*0.03=0.096       ...
#       [0, 1]:0.72      [0, 1]:0.xxx                 ...
#       [0, 2]:0.08      [0, 2]:0.xxx                 ...
#       ... 3(labels)^2(participants) = 9 states per step ...
#       [2, 2]:0.01      [2, 2]:0.xxx                 ...
#
# Note: More efficiently, we will store this as 3 ProParaDecoderState objects.
# Each ProParaDecoderState object has a group size of 9.
#
###################################
# States also store action history:
###################################
# action_hist:              time_step                    history of [p1_label, p2_label]
# action_history at state (step 1, row 1) from \phi   =  [[]]
# action_history at state (step 2,1 <- 1,1)           =  [[0,0]]
#       ... (step_num-1) action history elements for every state (9 per step * 3 steps) ...
# action_history at state (step 3,1 <- 2,1 <- 1,1)    =  [[0,0], [0,0]]
#
###################################
# Scoring each state:
###################################
# step 1, 1 => [0, 0]:0.32 = 0.8 (score of p1:label_0) * 0.4 (score of p2:label_0)
# To inject background, supplement with Prob(p1:label_0)*Prob(p2:label_0) = 0.8 * 0.1 = 0.08
#   that provides the general likelihood of something getting created or destroyed in this process.
# total_score at step 1,1 = 0.32 * 0.08 = 0.0256
#
from overrides import overrides



class ProParaDecoderState(DecoderState['ProParaDecoderState']):
    """
    Contains grouped states to be used in beam search, as described above.

    Parameters
    ----------
    group_indices: ``List[int]``
        One group consists of many states that are part of one step.
    action_history: ``List[List[List[int]]]``
        Has shape (group_size, num_steps, num_participants)
        e.g., assume a particular group
        [[0,0], [0,0]] = history at step 3,1 <- 2,1 <- 1,1
        here, <- step_num, label pair for p1, p2 = p1 label 0, p2 label 0
    participant_score_per_group: List[List[torch.autograd.Variable]]
        Has shape (group_size, num_participants)
    participants_embedding
        Has shape (group_size, num_participants, num_tokens * embedding_dim)
    logit_tensor:
        Has shape (group_size, num_participants)
        Supplies score: group_size, num_sentences, num_participants, label
        action logits.
        Though we know that it is less likely to have a DESTROY label
        in the opening step, we will let the model learn from the data.
        i.e. start value must be empty.
        To backprop, the flow of tensor from prostruct model cannot
        be broken (e.g., by pulling out scores like in action_logits)
        We will modify this logit_tensor all the way through.
    instance_id
        Required to update scores for a particular instance in the batch
        that the logit tensor represents.
    metadata, optional, default = {}
        Optional additional info (such as process_id)
        that the commonsense based scorer requires for topic specific lookup.
    overall_score: List[torch.autograd.Variable], optional (default = None)
        Score of the state (summed over all its participant scores)
    start_values: List[List[int]], optional (default = None)
    """

    def __init__(self,
                 group_indices: List[int],
                 action_history: List[List[List[int]]],
                 participant_score_per_group: List[List[torch.autograd.Variable]],
                 participants_embedding,
                 logit_tensor,
                 instance_id,
                 metadata={},
                 overall_score: List[torch.autograd.Variable] = None,
                 start_values: List[List[int]] = None) -> None:
        # To align with the Beam search code, we sum the scores of all participants.
        # This overall score of a state is also useful while sorting states.
        overall_score = overall_score or ProParaDecoderState.sum_participants_scores(participant_score_per_group)
        super().__init__(group_indices, action_history, overall_score)
        self.start_values = start_values or [0] * len(group_indices)
        self.participants_embedding = participants_embedding
        self.participant_score_per_group = participant_score_per_group
        self.logit_tensor = logit_tensor
        self.instance_id = instance_id
        self.metadata = metadata

    @classmethod
    def combine_states(cls, states) -> 'ProParaDecoderState':

        group_indices = [group_index for state in states for group_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        participant_scores_per_group = [score for state in states for score in state.participant_score_per_group]
        state_scores = [state_score for state in states for state_score in state.score]
        start_values = [start_value for state in states for start_value in state.start_values]

        return ProParaDecoderState(
            group_indices,
            action_histories,
            participant_scores_per_group,
            # All of the following (except start_values) are instance (not state) specific.
            states[0].participants_embedding,
            states[0].logit_tensor,
            states[0].instance_id,
            states[0].metadata,
            state_scores,
            start_values
        )

    def is_finished(self) -> bool:
        # Upon reaching the last step, the state is finished.
        # logit tensor has shape: (batch_size, num_sentences, num_participants, num_actions)
        max_steps = self.logit_tensor.shape[1]
        return len(self.action_history[0]) >= max_steps

    # Sum of the scores of the participants in this state.
    # This helps while sorting states based on "score" (that Beam search requires).
    # groupwise_score_per_participant shape: (num_group, num_steps, num_participants)
    @staticmethod
    def sum_participants_scores(groupwise_score_per_participant: List[List[torch.autograd.Variable]]) \
            -> List[torch.autograd.Variable]:
        groupwise_sum: List[torch.autograd.Variable] = []
        for group_index in range(0, len(groupwise_score_per_participant)):
            # shape: (num_participants,)
            participant_scores = groupwise_score_per_participant[group_index]
            # aggregate score of every participant in the current state.
            groupwise_sum.append(participant_scores.sum())
        return groupwise_sum

    # FIXME must depend on the history as well as scores.
    def __eq__(self, other):
        return isinstance(other, ProParaDecoderState) and self.action_history == other.action_history

    def __hash__(self):
        return hash(self.action_history)

    def __repr__(self):
        return f"{self.action_history} {self.score[0].data[0]}"

    def to_json(self):
        # used for prediction when the group/batchsize is 1
        # so only return the zeroth entry.
        return {"action_history": self.action_history[0], "score": self.score[0].data[0]}
