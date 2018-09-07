# This class reads ProPara grids in the following format
# and stores them in accessible data structures
#
#
# 4	SID	PARTICIPANTS	plants	bacteria	sediment	oil
# 4		PROMPT: How does oil form?	-=====	-=====	-=====	-=====
# 4	state1		?	?	?	-
# 4	event1	Plants die.
# 4	state2		?	?	?	-
# 4	event2	They are buried in sediment.
# 4	state3		sediment	?	?	-
# 4	event3	Bacteria is buried in the sediment.
# 4	state4		sediment	sediment	?	-
# 4	event4	Large amounts of sediment gradually pile on top of the original sediment.
# 4	state5		sediment	sediment	?	-
# 4	event5	Pressure builds up.
# 4	state6		sediment	sediment	?	-
# 4	event6	Heat increases.
# 4	state7		sediment	sediment	?	-
# 4	event7	The chemical structure of the buried sediment and plants changes.
# 4	state8		sediment	sediment	?	-
# 4	event8	The sediment and plants are at least one mile underground.
# 4	state9		one mile underground	sediment	underground	-
# 4	event9	The buried area is extremely hot.
# 4	state10		one mile underground	sediment	underground	-
# 4	event10	More chemical changes happen and the buried material becomes oil.
# 4	state11		-	-	underground	underground

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ProparaMetadata:
    def __init__(self, from_file_path: str):
        self.grids = dict()
        self.participants = dict()
        self.sentences = dict()
        self.prompts = dict()
        self.load(from_file_path)

    # Loads results from a file into a dictionary
    # process_id ->
    #               ques_id -> [answers]
    # where one answer is a tuple e.g.:
    # [init_value, final_value, loc_value, step_value]
    # Assuming input file contains=
    # processid TAB    quesid TAB   answer_tsv
    #   multiple answers separated by `tab`,
    #   slots within each answer separated by `++++`
    def load(self, from_file_path: str):
        logger.info("Propara metadata: Loading from: %s", from_file_path)

        with open(from_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                # process_id	State	PARTICIPANTS	plants	bacteria	sediment	oil
                # 4	            state1		             ?         	?	     ?	          -
                cols = line.split('\t')
                process_id = int(cols[0])
                if process_id not in self.grids.keys():
                    self.grids.setdefault(process_id, [])
                    self.participants.setdefault(process_id, [])
                    self.sentences.setdefault(process_id, [])
                if cols[2].lower() == "participants":
                    for c in cols[3:]:
                        if len(c) > 0:
                            self.participants[process_id] = cols[3:]
                elif cols[2].lower().startswith("prompt:"):
                    self.prompts[process_id] = cols[2].replace("PROMPT:", "").strip()
                elif cols[1].lower().startswith("event"):
                    self.sentences[process_id].append(cols[2])
                elif cols[1].lower().startswith("state"):
                    self.grids[process_id].append([c for c in cols[3:] if len(c) > 0])

    def get_sentences(self, para_id: int):
        return self.sentences[para_id]

    def get_participants(self, para_id: int):
        return self.participants[para_id]

    def get_para_ids(self):
        return self.prompts.keys()

    def get_prompt(self, para_id: int):
        return self.prompts[para_id]

    def get_grid(self, para_id: int):
        return self.grids[para_id]
