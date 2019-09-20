import csv
import itertools
import json


# Input:  grids format
#         14	SID	PARTICIPANTS	water	water vapor	droplets	rain	snow
#         14		PROMPT: What happens during the water cycle?	-=====	-=====	-=====	-=====	-=====
#         14	state1		ocean , lake , river , swamp , and plant	-	-	-	-
#         14	event1	Water from oceans, lakes, rivers, swamps, and plants turns into water vapor.
#         14	state2		-	cloud	-	-	-
#         14	event2	Water vapor forms droplets in clouds.
#         14	state3		-	-	cloud	-	-
#         14	event3	Water droplets in clouds become rain or snow and fall.
#         14	state4		-	-	-	ground	ground
#         14	event4	Some water goes into the ground.
#         14	state5		ground	-	-	-	-
#         14	event5	Some water flows down streams into rivers and oceans.
#         14	state6		river and ocean	-	-	-	-
#
# Output: json format
#         sentence_texts: List[str]
#         participants: List[str],
#         states: List[List[str]], where states[i][j] is ith participant at time j
#
# This class converts the tsv grids file format into json object per paragraph that can be passed to
# text_to_instance in ProParaDatasetReader


def convert_tsv_to_json(infile_path: str, outfile_path: str):
    out_file = open(outfile_path, "w")

    with open(infile_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for group_id, group in itertools.groupby(reader, lambda row: row and row[0]):
            if not group_id:
                continue

            participants = next(group)[3:]
            prompt = next(group)[2].split("PROMPT: ")[-1]

            # states[i][j] is the state of the i-th participant at time j
            states = [[] for _ in participants]

            sentence_texts = []

            for row in group:
                label = row[1]
                if label.startswith('state'):
                    # states start in the 4th column
                    for i, state in enumerate(row[3:]):
                        states[i].append(state)
                elif label.startswith('event'):
                    sentence_texts.append(row[2])
                else:
                    raise ValueError(f"unknown row type {label}")

            # output the paragraph into json format
            out_file.write("{")
            out_file.write("\"para_id\": " + json.dumps(group_id) + ", ")
            out_file.write("\"prompt\": " + json.dumps(prompt) + ", ")
            out_file.write("\"sentence_texts\": " + json.dumps(sentence_texts) + ", ")
            out_file.write("\"participants\": " + json.dumps(participants) + ", ")
            out_file.write("\"states\": " + json.dumps(states))
            out_file.write("}\n")
    out_file.close()

if __name__ == '__main__':
    convert_tsv_to_json(infile_path='../../data/emnlp18/grids.v1.train.tsv',
                        outfile_path='../../data/emnlp18/grids.v1.train.json')
    convert_tsv_to_json(infile_path='../../data/emnlp18/grids.v1.dev.tsv',
                        outfile_path='../../data/emnlp18/grids.v1.dev.json')
    convert_tsv_to_json(infile_path='../../data/emnlp18/grids.v1.test.tsv',
                        outfile_path='../../data/emnlp18/grids.v1.test.json')

