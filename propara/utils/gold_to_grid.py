from processes.utils.propara_metadata import ProparaMetadata

# Input: Gold ProPara grids from '../../tests/fixtures/decoder_data/grids-new-format.tsv'
# Output:  system output file
#         format: paraid \t stepid \t participant \t before_val \t after_val

paras = dict()
metadata_file = '../../tests/fixtures/decoder_data/grids-new-format.tsv'
propara = ProparaMetadata(metadata_file)


def create_grid_from_gold(output_filepath: str):
    # Start with a clean copy.
    paras.clear()
    out_file = open(output_filepath, "w")

    for process_id in propara.get_para_ids():
        sentences = propara.get_sentences(process_id)
        num_steps = len(sentences)
        participants = propara.get_participants(process_id)
        for participant_id in range(0, len(participants), 1):
            for step_id in range(1, num_steps+1, 1):
                before_val = propara.get_grid(process_id)[step_id-1][participant_id]
                after_val = propara.get_grid(process_id)[step_id][participant_id]
                action = 'NONE'
                if before_val == "-" and not after_val == '-':
                    action = 'CREATE'
                elif not before_val == "-" and after_val == '-':
                    action = 'DESTROY'
                elif not before_val == after_val:
                    action = 'MOVE'

                out_file.write('\t'.join([str(process_id), str(step_id),
                                          participants[participant_id],
                                          action, before_val, after_val]) + "\n")
    out_file.close()


if __name__ == '__main__':
    create_grid_from_gold('../../data/emnlp18/gold_grids.tsv')
