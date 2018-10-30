from math import exp

from processes.commonsense import model_output_loader
from processes.commonsense.background_knowledge.kb import KB


# To create the lexical kb: First: DeepEx over BUSC2, then, TestAI2Lexicon.test_endtoend
# (allennlp) nikett:processes nikett$ python -m processes.commands.run
# cmd_predict_scisrlv2 --archive_file
# /Users/nikett/quick/allennlp-models/scisrlv2/deepex22k-withelmo.tk_63iqn8wdimee.model.tar.gz
# --input_file ~/quick/busc2kb/inputs/
# --output_file ~/quick/busc2kb/outputs/deepex-v2.0-overall-prostruct-busc2.txt
# --output_format deepex


class KBLexical(KB):
    def __init__(self,
                 lexical_kb_path: str = "tests/fixtures/decoder_data/kbs/kb3/lexical-kb-v0.tsv",
                 fullgrid_prompts_load_path: str = "tests/fixtures/decoder_data/kbs/kb2/full-grids.tsv"
                 ):
        super().__init__()
        self.name = 'kb_lexical'
        # prompt -> process ids with that prompt
        self.prompts_to_processid = dict()
        # process_id -> prompt.
        self.processid_to_prompt = dict()
        self.load_prompts_fullgrids(fullgrid_prompts_load_path)
        # process_id -> knowledge tuples.
        self.kb = dict()
        self.load_kb(lexical_kb_path)

    def load_prompts_fullgrids(self, load_from):
        prompt_marker = "PROMPT:"
        with open(load_from, 'r') as infile:
            for line in infile:
                line = line.strip()
                cols = line.split("\t")
                if len(cols) > 3 and cols[2].startswith(prompt_marker):
                    prompt = cols[2].replace(prompt_marker, "").strip()
                    process_id = int(cols[0])
                    # Create space for prompt.
                    if prompt not in self.prompts_to_processid.keys():
                        self.prompts_to_processid.setdefault(prompt, [])
                    # Any line with prompts: is a unique line.
                    self.prompts_to_processid[prompt].append(process_id)
                    self.processid_to_prompt[process_id] = prompt

    def prob_of(self, process_id, cand):
        cand_entity = cand[model_output_loader.IDX_ENTITY]
        cand_init_val = cand[model_output_loader.IDX_INITVAL]
        cand_final_val = cand[model_output_loader.IDX_FINALVAL]
        cand_eventtype = cand[model_output_loader.IDX_EVENTTYPE]
        cand_score = cand[model_output_loader.IDX_SCORE]
        prompt = self.processid_to_prompt.get(process_id, "")
        # we cannot say about any step where NONE is predicted.
        if (cand_eventtype == "NONE"
            or prompt not in self.kb
            # Ideally in embedding space not hardcoded.
            or cand_eventtype.upper() not in self.kb[prompt]):
            return 1.0

        # Take logistic function (L / 1 + e -k(x - x0))
        # x0 : center point (where value = 0.5)
        # k  : steepness (more steep means more quickly picks from x0 to max L)
        # We set: k = 1, x0 = 2.0, L = 1.0
        bk_freq = self.kb[prompt][cand_eventtype].get(cand_entity, 0.0)
        # if bk_freq >= 0:
        #     print(f"DEBUG: verifying {cand} from lexical_bk to get freq: {bk_freq}")
        return 1.0 / (1 + exp(2.0 - bk_freq))

    def load_kb(self, lexical_kb_path):
        """
        :param lexical_kb_path:
        Contains tsv entries (topic, triple, sent)
                 How are proteins synthesized in a cell?.tsv     
                 produce:energy:CREATED  
                 The mitochondria are the place where your cells produce the energy they need from the nutrients in the food you eat.
        :return:
         {
            prompt -> {
                            CREATED -> energy (freq), ...    
                            DESTROYED ->   
                            MOVED ->   
                          }
         }
         Note: Ideally we need to embed the commonsense.
               to generalize better, esp. with unseen topics at test time.
        """
        appendix = ".tsv"
        with open(lexical_kb_path, 'r') as infile:
            for line in infile:
                line = line.strip()
                cols = line.split("\t")
                if len(cols) >= 2:  # Atleast topic and triples.
                    prompt = cols[0].replace(appendix, "").strip()
                    # Create space for prompt.
                    if prompt not in self.kb.keys():
                        self.kb.setdefault(prompt, {})
                    # produce:energy:CREATED
                    verb, entity, label = cols[1].strip().split(":")
                    label = (label
                             .replace("CREATED", "CREATE")
                             .replace("DESTROYED", "DESTROY")
                             .replace("MOVED", "MOVE"))
                    if label not in self.kb[prompt].keys():
                        self.kb[prompt].setdefault(label, {})
                    # Add to dict of interest or increment it's existing value.
                    # e.g., DESTROYED -> {energy: 20, food: 10}
                    #         update to: {energy: 20, food: 11}
                    d = self.kb[prompt][label]
                    curr_freq = d.get(entity, 0) + 1
                    d[entity] = curr_freq
