from collections import OrderedDict
import os
import sys

import argparse

from propara.utils.propara_metadata import ProparaMetadata
from propara.eval.emnlp18_eval import Evaluator

# Input:  system output file
#         format: paraid \t stepid \t participant \t before_location_val \t after_location_val
# Output: processid \t quesid \t answer_tsv
#        (multiple answers separated by `\t`,
#         slots within each answer separated by `++++`,
#         conjunctions within a slot separated by ' AND '
#         synonyms separated by 'OR')
#
# This class converts the system generated state information into answers to set of
# predetermined questions for a given process
#
# Question-set:
#  Q1: What are the inputs?
#  Q2: What are the outputs?
#  Q3: What is converted into what?
#      tuple: (participant-list-from, participant-list-to, loc-list, step-id)
#  Q4: What is moved?
#      tuple: (participant, from-loc, to-loc, step-id)

# process_id => dict (participant -> state_values)
state_values = dict()
actions = dict()

oldSynonymSep = '; '
newSynonymSep = ' OR '
conjunctionSep = ' AND '
slotSep = '++++'
tupleSep = '\t'

if os.path.exists('../../tests/fixtures/emnlp18_eval/grids-new-format.tsv'):
    metadata_file = '../../tests/fixtures/emnlp18_eval/grids-new-format.tsv'
else:
    # pytest from CLI requires tests/fixtures/... path
    metadata_file = 'tests/fixtures/emnlp18_eval/grids-new-format.tsv'
propara = ProparaMetadata(metadata_file)


def load(input_filepath: str):
    # Start with a clean copy.
    state_values.clear()
    actions.clear()

    with open(input_filepath) as f:
        for line in f:
            cols = line.strip().split('\t')
            assert len(cols) >= 6
            # paraid \t stepid \t participant \t action \t before_val \t after_val \t gold_before_val \t gold_after_val
            process_id = int(cols[0])
            if process_id not in state_values:
                state_values.setdefault(process_id, OrderedDict())
                actions.setdefault(process_id, OrderedDict())

            step_id = int(cols[1])
            participant = cols[2]
            action = cols[3]
            before_val = cols[4]
            if before_val == "-":
                before_val = "null"
            elif before_val == "?":
                before_val = "unk"

            after_val = cols[5]
            if after_val == "-":
                after_val = "null"
            elif after_val == "?":
                after_val = "unk"

            num_steps = len(propara.get_sentences(process_id))
            num_states = num_steps + 1
            existing_participant_states = state_values.get(process_id, OrderedDict())
            existing_participant_actions = actions.get(process_id, OrderedDict())

            if participant not in existing_participant_states.keys():
                existing_participant_states.setdefault(participant, ['unk']*num_states)
                existing_participant_actions.setdefault(participant, ['NONE']*num_steps)
            existing_state_values = existing_participant_states[participant]
            existing_actions = existing_participant_actions[participant]

            if step_id == 1:
                existing_state_values[step_id-1] = before_val
            existing_state_values[step_id] = after_val
            existing_actions[step_id-1] = action
    return state_values


def all_process_ids():
    return state_values.keys()


def get_para_state_values(process_id: int):
    return state_values.get(process_id, dict())


def get_para_actions(process_id: int):
    return actions.get(process_id, dict())


#  Q1: What are the inputs?
#      - If a participant exists in state1, but does not exist in the end stateN, it's an input.
def answer_q1(process_id: int):
    participant_dict = get_para_state_values(process_id)
    participant_actions_dict = get_para_actions(process_id)
    inputs = []

    for participant in participant_dict.keys():
        # state_values = participant_dict[participant]
        actions = participant_actions_dict[participant]

        is_input = is_this_action_seq_of_an_input(actions)
        if is_input:
            inputs.append(newSynonymSep.join(participant.split(oldSynonymSep)))
        # num_states = len(state_values)
        # if state_values[0] != "null" and state_values[num_states-1] == "null":
        #     inputs.append(newSynonymSep.join(participant.split(oldSynonymSep)))
    return inputs


def is_this_action_seq_of_an_input(actions) -> bool:
    for action_id, action in enumerate(actions):
        no_create_before = 'CREATE' not in actions[0:action_id-1]
        current_destroy = actions[action_id] == 'DESTROY'
        no_create_move_later = 'CREATE' not in actions[action_id+1:] \
                          and 'MOVE' not in actions[action_id + 1:]
                          #and 'DESTROY' not in actions[action_id + 1:] \

        if no_create_before and current_destroy and no_create_move_later:
            return True
    return False


#  Q2: What are the outputs?
#      - If a participant does not exist in state1, but exists in the end stateN, it's an output.
def answer_q2(process_id: int):
    participant_dict = get_para_state_values(process_id)
    participant_actions_dict = get_para_actions(process_id)
    outputs = []

    for participant in participant_dict.keys():
        # state_values = participant_dict[participant]
        actions = participant_actions_dict[participant]

        is_output = is_this_action_seq_of_an_output(actions)
        if is_output:
            outputs.append(newSynonymSep.join(participant.split(oldSynonymSep)))
            # num_states = len(state_values)
            # if state_values[0] != "null" and state_values[num_states-1] == "null":
            #     inputs.append(newSynonymSep.join(participant.split(oldSynonymSep)))
    return outputs


def is_this_action_seq_of_an_output(actions) -> bool:
    for action_id, action in enumerate(actions):
        no_destroy_move_before = 'DESTROY' not in actions[0:action_id - 1] and 'MOVE' not in actions[0:action_id - 1]
        current_create = actions[action_id] == 'CREATE'
        no_destroy_later = 'DESTROY' not in actions[action_id + 1:]
        if no_destroy_move_before and current_create and no_destroy_later:
            return True
    return False


#  Q3: What is converted?
#      tuple: (participant-list-from, participant-list-to, loc-list, step-id)
#      a. For any event with BOTH "D" and "C" in:
#       	The "D" participants are converted to the "C" participants at the union of the D and C locations
#      b. IF an event has ONLY "D" but no "C" in   ("M" is ok - irrelevant)
#       	AND the NEXT event has ONLY "C" but no "D" in   ("M" is ok - irrelevant)
#       	THEN the "D" participants are converted to the "C" participants at the union of the D and C locations
def answer_q3(process_id: int):
    conversions = []
    num_steps = len(propara.get_sentences(process_id))
    for step_id in range(1, num_steps+1, 1):
        (created, c_locations) = get_created_at_step(process_id, step_id)
        (destroyed, d_locations) = get_destroyed_at_step(process_id, step_id)
        if created and destroyed:
            conversions.append(slotSep.join((conjunctionSep.join(destroyed), conjunctionSep.join(created), conjunctionSep.join(set(c_locations+d_locations)), str(step_id))))
        elif destroyed and step_id < num_steps-1:
            (created2, c_locations2) = get_created_at_step(process_id, step_id+1)
            (destroyed2, d_locations2) = get_destroyed_at_step(process_id, step_id+1)
            created_but_not_destroyed = set(created2) - set(destroyed)
            if not destroyed2 and created_but_not_destroyed:
                conversions.append(slotSep.join((conjunctionSep.join(destroyed), conjunctionSep.join(created_but_not_destroyed), conjunctionSep.join(set(c_locations2 + d_locations)), str(step_id))))
        elif created and step_id < num_steps-1:
            (created2, c_locations2) = get_created_at_step(process_id, step_id+1)
            (destroyed2, d_locations2) = get_destroyed_at_step(process_id, step_id+1)
            destroyed_but_not_created = set(destroyed2) - set(created)
            if not created2 and destroyed_but_not_created:
                conversions.append(slotSep.join((conjunctionSep.join(destroyed_but_not_created), conjunctionSep.join(created), conjunctionSep.join(set(c_locations + d_locations2)), str(step_id))))

    return conversions


#  Q4: What is moved?
#      tuple: (participant, from-loc, to-loc, step-id)
#  return all moves
def answer_q4(process_id: int):
    participant_state_values = get_para_state_values(process_id)
    participant_actions = get_para_actions(process_id)

    moved = []
    for participant in participant_state_values.keys():
        state_values = participant_state_values[participant]
        actions = participant_actions[participant]

        num_states = len(state_values)

        for step_id in range(1, num_states, 1):
            if is_moved(state_values, actions, step_id):
                moved.append(slotSep.join((newSynonymSep.join(participant.split(oldSynonymSep)), state_values[step_id-1], state_values[step_id], str(step_id))))
    return moved


def is_moved(state_values: [], actions: [], step_id: int):
    if actions[step_id-1] == "MOVE":
        return True
    if state_values[step_id-1] != "null" and state_values[step_id] != "null" and state_values[step_id-1] != state_values[step_id]:
        return True
    return False


def get_created_at_step(process_id: int, step_id: int):
    participant_dict = get_para_state_values(process_id)
    created = []
    locations = []

    for participant in participant_dict.keys():
        state_values = participant_dict[participant]
        if is_creation(state_values, step_id):
            created.append(newSynonymSep.join(participant.split(oldSynonymSep)))
            locations.append(state_values[step_id])

    return created, locations


def get_destroyed_at_step(process_id: int, step_id: int):
    participant_dict = get_para_state_values(process_id)
    destroyed = []
    locations = []

    for participant in participant_dict.keys():
        state_values = participant_dict[participant]
        if is_destruction(state_values, step_id):
            destroyed.append(newSynonymSep.join(participant.split(oldSynonymSep)))
            locations.append(state_values[step_id-1])

    return destroyed, locations


def is_creation(state_values: [], step_id):
    if state_values[step_id-1] == "null" and state_values[step_id] != "null":
        return True
    return False


def is_destruction(state_values: [], step_id):
    if state_values[step_id-1] != "null" and state_values[step_id] == "null":
        return True
    return False


def read_grid_input(
        infile_path='data/emnlp18/prostruct.pred.test.tsv',
        outfile_path='data/emnlp18/prostruct.pred.test.qa.tsv'):

    print(f"\nReading model predictions from: {infile_path}")
    load(input_filepath=str(infile_path))

    out_file = open(outfile_path, "w")

    for process_id, contents in state_values.items():
        # output record format
        # processid \t quesid \t answer_tsv
        # (multiple answers in answer_tsv are separated by `tab`, and the  slots within each answer separated by `++++`)
        out_file.write(tupleSep.join([str(process_id), "1", '\t'.join(answer_q1(process_id))]) + "\n")
        out_file.write(tupleSep.join([str(process_id), "2", '\t'.join(answer_q2(process_id))]) + "\n")
        out_file.write(tupleSep.join([str(process_id), "3", '\t'.join(answer_q3(process_id))]) + "\n")
        out_file.write(tupleSep.join([str(process_id), "4", '\t'.join(answer_q4(process_id))]) + "\n")

    print(f"Derived QA from model predictions in: {outfile_path}")
    out_file.close()

# For sanity check, the output to the followingshould be 100.0 F1
# python propara/utils/end2end_grid_to_qa.py --predictions data/emnlp18/prostruct.pred.test.tsv --path_to_store_derived_qa /tmp/sanity.tsv --testset_path data/emnlp18/prostruct.pred.test.qa.tsv
#
# Expected output:
# Reading model predictions from: data/emnlp18/prostruct.pred.test.tsv
# Derived QA from model predictions in: /tmp/sanity.tsv
# =======================================================================
#
# Average Precision/Recall/F1 =  100.0	100.0	100.0
#
# =======================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation script for ProStruct [https://arxiv.org/abs/1808.10012].',
                                     usage="\tpython utils/end2end_grid_to_qa.py "
                                           "\n\t\t--predictions /path/to/predictions.tsv"
                                           "\n\t\t--path_to_store_derived_qa /tmp/derived_qa.tsv"
                                           "\n\t\t--testset_path /path/to/gold.tsv")

    parser.add_argument('--predictions',
                        action='store',
                        dest='predictions',
                        required=True,
                        help='Path to the state-change grids predicted by the model, '
                             'the expected format of this tsv file is: \n'
                             '(para_id_from_test_partition\tstep_id_starts_from_1\tentity\tstate_change\tfrom_loc\tto_loc)'
                             ' e.g., 310	1	blood	MOVE	?	arteries.\n')

    parser.add_argument('--path_to_store_derived_qa',
                        action='store',
                        dest='path_to_store_derived_qa',
                        required=True,
                        help='Using the predictions (predicted state-change grid) derive answers to 4 QAs '
                             'in ProStruct paper. These will be evaluated against testset. Format produced: '
                             'para_id\tques_id\tanswer_tsv')

    parser.add_argument('--testset_path',
                        action='store',
                        dest='testset_path',
                        required=True,
                        help='The testset containing'
                             'para_id\tques_id\texpected_answer')

    args = parser.parse_args()
    read_grid_input(infile_path=args.predictions,
                    outfile_path=args.path_to_store_derived_qa)

    # Evaluation code to get question wise and average F1 scores
    evaluator = Evaluator(system_file=args.path_to_store_derived_qa,
                          gold_file=args.testset_path)
    metrics_all = evaluator.score_all_questions()
    # print(evaluator.pretty_print(metrics_all))
    q_wise = evaluator.ques_wise_overall_metric(metrics_all)
    # print(q_wise)
    # print(round(sum(q_wise) / len(q_wise), 3))

    # Compute average Precision/Recall/F1 scores
    q_wise_metrics = []
    overall_PRF = []
    for metric_id in range(1, 3, 1):
        q_wise_PRF = evaluator.ques_wise_overall_metric(metrics_all, metric_id)
        # print('metric_id:', metric_id)
        q_wise_metrics.append(q_wise_PRF)
        # print("q_wise_PRF:", q_wise_PRF)
        # print(round(sum(q_wise_PRF) / len(q_wise_PRF), 3))
        overall_PRF.append(str(round(sum(q_wise_PRF) / len(q_wise_PRF), 4) * 100.0))

    # Compute average F1 per question category
    average_f1 = 0.0
    for q_id in range(4):
        precision_qid = q_wise_metrics[0][q_id]
        recall_qid = q_wise_metrics[1][q_id]
        f1_qid = round((2 * precision_qid * recall_qid) / (precision_qid + recall_qid), 4)
        # print(f"F1 for qid:{q_id} = {f1_qid}")
        average_f1 += f1_qid
    average_f1 = average_f1 / 4.0

    # Print Precision/Recall/F1 scores averaged across all questions
    overall_P = float(overall_PRF[0])
    overall_R = float(overall_PRF[1])
    overall_F1 = round((2 * overall_P * overall_R) / (overall_P + overall_R), 2)
    overall_PRF.append(str(overall_F1))
    print("=======================================================================")
    print("\nAverage Precision/Recall/F1 = ", '\t'.join(overall_PRF), "\n")
    print("=======================================================================")
