from collections import OrderedDict

IDX_ENTITY = 0
IDX_EVENTTYPE = 1
IDX_INITVAL = 2
IDX_FINALVAL = 3
IDX_SCORE = 4
EXPECTED_LEN_CAND = 5

# process_id => dict (step_id -> list of tuples)
paras = dict()


# Beam of size=4 should require top-4 candidate events
# per step in a process paragraph
# processid stepid  participant eventtype   initval    finalval prolocal_confidence
# 514    1    snow    MOVE    ?    area    0.691445351
# 514    1    snow    DESTROY    ?    -    0.308446348
# 514    1    snow    NONE    area    area    1.07E-04
# 514    1    snow    CREATE    -    area    1.08E-06
def load(input_filepath):
    # Start with a clean copy.
    paras.clear()
    with open(input_filepath) as f:
        for line in f:
            if not line.strip().startswith('#'):
                cols = line.strip().split('\t')
                assert len(cols) == 7
                process_id = int(cols[0])
                # Process has a bunch of step ids.
                if process_id not in paras:
                    paras.setdefault(process_id, OrderedDict())  # paras.setdefault(process_id, [])
                step_id = int(cols[1])
                existing_steps = paras.get(process_id, OrderedDict())
                # Step has a bunch of predictions.
                if step_id not in existing_steps.keys():
                    existing_steps.setdefault(step_id, [])
                existing_step_values = existing_steps[step_id]
                # Now, add values to this step.
                val_of_step = (cols[2], cols[3], cols[4], cols[5], float(cols[6]))
                existing_step_values.append(val_of_step)


def all_process_ids():
    return paras.keys()


def get_para(process_id: int):
    return paras.get(process_id, dict())


# FIXME: these are not in order.
def step_ids(process_id: int):
    return list(get_para(process_id).keys())


def step_id_at_pos(process_id: int, pos):
    return step_ids(process_id)[pos]


def step_candidates(process_id, step_id):
    return get_para(process_id).get(step_id, [])


# Returns tuple such as: (snow, MOVE, ?, area, 0.691)
def step_cand(process_id, step_id, cand_pos_index):
    candidates = step_candidates(process_id, step_id)
    if len(candidates) > cand_pos_index:
        return candidates[cand_pos_index]
    else:
        return ()


def num_candidates(process_id, step_id):
    return len(step_cand(process_id, step_id))


def usage():
    infile_path = '../../tests/fixtures/decoder_data/sample_prolocal_beam.tsv'
    print("Reading from filepath: ", infile_path)
    load(infile_path)
    for k, v in paras.items():
        print("\n\n" + str(k) + "\n-----------------------------\n")
        for k2, v2 in v.items():
            print(k2, v2)
    print("\n\nFetching 2nd step from para 514")
    print(step_candidates(514, 2))
    print("\n\nFetching 1st step from para 5141")
    print(step_candidates(5141, 1))
    print("\n\nFetching 2nd step, 1st cand from para 514")
    print(step_cand(514, 2, 1))
    print("\n\nStep ids in 514 are ordered?")
    print(step_ids(514))
    print(step_id_at_pos(514, 0))
    print(step_id_at_pos(514, 1))
    print(step_id_at_pos(514, 2))

if __name__ == '__main__':
    usage()
