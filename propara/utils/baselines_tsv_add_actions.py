from collections import OrderedDict
import os
import sys
import json

# Input:  Grid in TSV format with following format (action column is missing)
#         paraid \t stepid \t participant \t from_location \t to_location
# Output: Grid in TSV format with Action column derived from from_location and to_location
#         paraid \t stepid \t participant \t action \t from_location \t to_location
#
# This class converts the propara baseline generated predictions file and adds a column for "action"
# based on predicted from-location and to-location.


def load(input_filepath: str,
         output_filepath: str):

    out_file = open(output_filepath, "w")

    with open(input_filepath) as f:
        for line in f:
            cols = line.strip().split('\t')
            assert len(cols) == 5 or len(cols) == 7
            # paraid \t stepid \t participant \t before_val \t after_val
            # or
            # paraid \t stepid \t participant \t before_val \t after_val  \t gold_before_val  \t  gold_after_val
            process_id = int(cols[0])

            step_id = int(cols[1])
            participant = cols[2]
            before_val = cols[3]
            if before_val == "null":
                before_val = "-"
            elif before_val == "unk":
                before_val = "?"

            after_val = cols[4]
            if after_val == "null":
                after_val = "-"
            elif after_val == "unk":
                after_val = "?"

            if before_val == "-" and after_val != "-":
                action = 'CREATE'
            elif before_val != "-" and after_val == "-":
                action = 'DESTROY'
            elif before_val != "-" and after_val != "-" and before_val != after_val:
                action = 'MOVE'
            else:
                action = 'NONE'

            out_file.write('\t'.join([str(process_id), str(step_id), participant, action, before_val, after_val]) + '\n')

    out_file.close()


def read_grid_input(infile_path, outfile_path):
    print("Reading from filepath: ", infile_path)

    load(input_filepath=str(infile_path),
         output_filepath=str(outfile_path))

if __name__ == '__main__':
    indir = "/tmp/baseline/"
    read_grid_input(
        infile_path=indir+'/pred.test.tsv',
        outfile_path=indir+'/pred.test.withActions.tsv')