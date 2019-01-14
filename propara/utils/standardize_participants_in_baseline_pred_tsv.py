from collections import OrderedDict
import os
import sys

from propara.evaluation.parti2id import Parti2ID
from propara.utils.propara_metadata import ProparaMetadata
from propara.evaluation.eval import Evaluator
import json

# Input:  TSV grid in the following format
#         paraid \t stepid \t participant \t action \t from_location \t to_location
# Output: TSV grid with participant column standardized w.r.t the gold grids
#         paraid \t stepid \t standardized-participant \t action \t from_location \t to_location
#

state_values = dict()
actions = dict()

oldSynonymSep = '; '
newSynonymSep = ' OR '
conjunctionSep = ' AND '
slotSep = '++++'
tupleSep = '\t'

if os.path.exists('tests/fixtures/decoder_data/grids-new-format.tsv'):
    metadata_file = 'tests/fixtures/decoder_data/grids-new-format.tsv'
else:
    # pytest from CLI requires tests/fixtures/... path
    metadata_file = '../../tests/fixtures/decoder_data/grids-new-format.tsv'
propara = ProparaMetadata(metadata_file)
participant_to_id_converter = Parti2ID()

def load(input_filepath: str,
         output_filepath: str):
    # Start with a clean copy.
    state_values.clear()
    actions.clear()
    out_file = open(output_filepath, "w")
    prev_process_id = -100
    outputted_record_keys = set()
    outputted_participants = set()

    with open(input_filepath) as f:
        for line in f:
            cols = line.strip().split('\t')
            assert len(cols) == 6
            # paraid \t stepid \t participant \t action \t before_val \t after_val

            process_id = int(cols[0])
            if prev_process_id != process_id:
                if prev_process_id >= 0:
                    # output unrecovered participants from Gold with actions for each step = NONE and both locations = ?
                    prev_gold_participants = propara.get_participants(prev_process_id)
                    prev_num_steps = len(propara.get_sentences(prev_process_id))

                    for p in prev_gold_participants:
                        if p not in outputted_participants:
                            for s_id in range(prev_num_steps):
                                standardized_record = '\t'.join(
                                    [str(prev_process_id), str(s_id+1), p, 'NONE', '?', '?']) + '\n'
                                out_file.write(standardized_record)

                prev_process_id = process_id
                outputted_record_keys = set()
                outputted_participants = set()

            step_id = int(cols[1])
            predicted_participant = cols[2]
            action = cols[3]
            before_val = cols[4]
            if before_val == "null":
                before_val = "-"
            elif before_val == "unk":
                before_val = "?"

            after_val = cols[5]
            if after_val == "null":
                after_val = "-"
            elif after_val == "unk":
                after_val = "?"

            participants = propara.get_participants(process_id)
            participants_dict = {}
            p_id = 0
            for p in participants:
                p_id += 1
                participants_dict[p] = p_id

            matched_participants = participant_to_id_converter.best_matching_id(
                q=predicted_participant,
                p_map=participants_dict)

            if matched_participants:
                std_participant = matched_participants[participant_to_id_converter.literal_matched_parti_str]
                new_record_key = '\t'.join([str(process_id), str(step_id), std_participant])
                if new_record_key not in outputted_record_keys:
                    standardized_record = '\t'.join(
                        [str(process_id), str(step_id), std_participant, action, before_val, after_val]) + '\n'
                    outputted_participants.add(std_participant)
                    outputted_record_keys.add(new_record_key)
                    out_file.write(standardized_record)

    if prev_process_id >= 0:
        # output unrecovered participants from Gold with actions for each step = NONE and both locations = ?
        prev_gold_participants = propara.get_participants(prev_process_id)
        prev_num_steps = len(propara.get_sentences(prev_process_id))

        for p in prev_gold_participants:
            if p not in outputted_participants:
                for s_id in range(prev_num_steps):
                    standardized_record = '\t'.join(
                        [str(prev_process_id), str(s_id + 1), p, 'NONE', '?', '?']) + '\n'
                    out_file.write(standardized_record)

    out_file.close()
    return state_values


def read_grid_input(infile_path, outfile_path):
    print("Reading from filepath: ", infile_path)

    load(input_filepath=str(infile_path),
         output_filepath=str(outfile_path))

if __name__ == '__main__':
    indir = "/tmp/proglobal/"
    read_grid_input(
        infile_path=indir+'/proglobal.test.pred.tsv',
        outfile_path=indir+'/proglobal.test.pred.stdParticipants.tsv'
    )