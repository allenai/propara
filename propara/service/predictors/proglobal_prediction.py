from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('ProGlobalPrediction')
class ProGlobalPredictor(Predictor):
    """
    The predictor function for ProGlobal:
        it will read each line of the Json file and convert it to an instance for ProGlobal
        the output will contain: para_id, entity, paragraph, best_span (system), true_span (gold)
    """
    
    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance_text = inputs["instance"]
        step_list = instance_text.split("####")
        tokenizer = JustSpacesWordSplitter()

        headline = step_list[0]

        parts = headline.strip().split('\t')
        para_id = parts[0]
        participant = parts[1]
        step_count = int(parts[2])

        paragraph = "";

        sents_list = []  # sentences of each paragraph
        sents_anno_list = []  # list of sentence annotations (P C F)
        word_pos_list = []  # list of word pos, -2 -1 0 1 2, to indicate the position of participants
        part_mask_list = []  # list of participant mask, 0 0 1 1 0 0, to obtain the part embeddings
        before_category_status_list = []  # list of bef category annotations   0-known, 1-unknown, 2-null
        before_category_mask_list = []  # list of bef category masks, 0-unknown   1-known
        before_loc_start_list = []  # list of start positions of bef loc
        before_loc_end_list = []  # list of end positions of bef loc
        after_category_status_list = []  # list of aft category annotations   0-known, 1-unknown, 2-null
        after_category_mask_list = []  # list of aft category masks, 0-unknown   1-known
        after_loc_start_list = []  # list of start positions of aft loc
        after_loc_end_list = []  # list of end positions of aft loc

        i = 1
        while i<len(step_list):
            para_line = step_list[i]
            para_line = para_line.lower()
            words = tokenizer.split_words(para_line.strip())

            paragraph = para_line

            i = i+1
            sent_anno_line = step_list[i]
            sent_annos = tokenizer.split_words(sent_anno_line.strip())

            i = i+1
            anno_line = step_list[i]
            anno_parts = anno_line.split('\t')
            participant = anno_parts[0].lower().split()
            part_start = int(anno_parts[1])
            part_end = int(anno_parts[2])

            i = i+1

            participant_mask = []

            before_loc = anno_parts[3].lower().split()
            before_loc_start = int(anno_parts[4])
            before_loc_end = int(anno_parts[5])
            after_loc = anno_parts[7].lower().split()
            after_loc_start = int(anno_parts[8])
            after_loc_end = int(anno_parts[9])

            if before_loc_start == -3:
                before_loc = "null"
                before_loc_start = -2
                before_loc_end = -2
            if after_loc_start == -3:
                after_loc = "null"
                after_loc_start = -2
                after_loc_end = -2

            word_pos_line = ""
            for m in range(0, part_start):
                pos = m - part_start
                word_pos_line += " " + str(pos)
                participant_mask.append(0)
            for m in range(part_start, part_end):
                pos = 0
                word_pos_line += " " + str(pos)
                participant_mask.append(1)
            for m in range(part_end, len(words)):
                pos = m - part_end
                word_pos_line += " " + str(pos)
                participant_mask.append(0)

            input_length = len(words)
            word_pos = tokenizer.split_words(word_pos_line.strip())

            assert input_length == len(participant_mask)

            # 0: known    1: unk     2: null
            before_category_status = 0
            after_category_status = 0

            # 1: known     0: category
            before_category_mask = 1
            after_category_mask = 1

            category_index = 0
            # -2 -- null,   -1 -- unk
            if before_loc_start == -2 and before_loc_end == -2:
                before_category_status = 2
                before_category_mask = 0
                before_loc_start = category_index
                before_loc_end = category_index
            elif before_loc_start == -1 and before_loc_end == -1:
                before_category_status = 1
                before_category_mask = 0
                before_loc_start = category_index
                before_loc_end = category_index
            if after_loc_start == -2 and after_loc_end == -2:
                after_category_status = 2
                after_category_mask = 0
                after_loc_start = category_index
                after_loc_end = category_index
            elif after_loc_start == -1 and after_loc_end == -1:
                after_category_status = 1
                after_category_mask = 0
                after_loc_start = category_index
                after_loc_end = category_index

            sents_list.append(words)
            sents_anno_list.append(sent_annos)
            word_pos_list.append(word_pos)
            part_mask_list.append(participant_mask)
            before_category_status_list.append(before_category_status)
            before_category_mask_list.append(before_category_mask)
            before_loc_start_list.append(before_loc_start)
            before_loc_end_list.append(before_loc_end)
            after_category_status_list.append(after_category_status)
            after_category_mask_list.append(after_category_mask)
            after_loc_start_list.append(after_loc_start)
            after_loc_end_list.append(after_loc_end)
        instance = self._dataset_reader.text_to_instance([sents_list, sents_anno_list, word_pos_list, part_mask_list,
                                                        before_category_status_list, before_category_mask_list,
                                                        before_loc_start_list, before_loc_end_list,
                                                        after_category_status_list, after_category_mask_list,
                                                        after_loc_start_list, after_loc_end_list])

        outputs = self._model.forward_on_instance(instance, cuda_device)

        predictions = {}
        predictions["paraid"] = para_id
        predictions["entity"] = parts[1]
        predictions["paragraph"] = paragraph
        predictions["best_span"] = str(outputs["best_span"].numpy())
        predictions["true_span"] = str(outputs["true_span"].numpy())

        return predictions