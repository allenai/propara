from typing import Dict, List

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.instance import Instance
from allennlp.data.fields.field import Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, IndexField, LabelField, ListField, SequenceLabelField
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

@DatasetReader.register("ProGlobalDatasetReader")
class ProGlobalDatasetReader(DatasetReader):
    """
    Reads a file from ProPara state inference dataset. Each instance contains one participant and multiple steps.
    Format:
        ParaID \t Participant \t TotalSteps
        Step1-Paragraph
        Step1-Sent Indicator (P/C/F in terms of each word in Step1-Paragraph)
        Step1-Annotation (Participant \t Participant_Start \t Participant_end \t loc_bef \t loc_before_start
        \t loc_before_end \t loc_aft \t loc_after_start \t loc_after_end)
        Step2 ...

        We convert each instance into fields named
        "tokens_list"  "positions_list"  "sent_positions_list"  "before_loc_start"  "before_loc_end"
        "after_loc_start_list"  "after_loc_end_list"  "before_category"  "after_category_list"
        "before_category_mask"  "after_category_mask_list"

    Parameters
    ------------
    token_indexers: Dict[str, TokenIndexer], required
    token_position_indexers: Dict[str, TokenIndexer], required
    sent_position_indexers: Dict[str, TokenIndexer], required
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 token_position_indexers: Dict[str, TokenIndexer] = None,
                 sent_position_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._token_indexers = token_indexers or {'tokens_list': SingleIdTokenIndexer()}
        self._token_position_indexers = token_position_indexers or {'token_positions_list': SingleIdTokenIndexer()}
        self._sent_position_indexers = sent_position_indexers or {'sent_positions_list': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        tokenizer = JustSpacesWordSplitter()

        with open(file_path, 'r') as f:
            while True:
                headline = f.readline()
                if not headline:
                    break
                parts = headline.strip().split('\t')
                step_count = int(parts[2])

                sents_list = []        # sentences of each paragraph
                sents_anno_list = []   # list of sentence annotations (P C F)
                word_pos_list = []     # list of word pos, -2 -1 0 1 2, to indicate the position of participants
                part_mask_list = []    # list of participant mask, 0 0 1 1 0 0, to obtain the part embeddings
                before_category_status_list = []   # list of bef category annotations   0-known, 1-unknown, 2-null
                before_category_mask_list = []     # list of bef category masks, 0-unknown   1-known
                before_loc_start_list = []      # list of start positions of bef loc
                before_loc_end_list = []        # list of end positions of bef loc
                after_category_status_list = []   # list of aft category annotations   0-known, 1-unknown, 2-null
                after_category_mask_list = []     # list of aft category masks, 0-unknown   1-known
                after_loc_start_list = []      # list of start positions of aft loc
                after_loc_end_list = []        # list of end positions of aft loc

                for i in range(step_count):
                    paraline = f.readline()
                    paraline = paraline.lower()
                    words = tokenizer.split_words(paraline.strip())

                    sent_anno_line = f.readline()
                    sent_annos = tokenizer.split_words(sent_anno_line.strip())

                    anno_line = f.readline()
                    anno_parts = anno_line.split('\t')
                    part_start = int(anno_parts[1])
                    part_end = int(anno_parts[2])

                    participant_mask = []

                    before_loc = anno_parts[3].lower().split()
                    before_loc_start = int(anno_parts[4])
                    before_loc_end = int(anno_parts[5])
                    after_loc = anno_parts[7].lower().split()
                    after_loc_start = int(anno_parts[8])
                    after_loc_end = int(anno_parts[9])

                    word_pos_line = ""
                    for i in range(0, part_start):
                        pos = i - part_start
                        word_pos_line += " " + str(pos)
                        participant_mask.append(0)
                    for i in range(part_start, part_end):
                        pos = 0
                        word_pos_line += " " + str(pos)
                        participant_mask.append(1)
                    for i in range(part_end, len(words)):
                        pos = i - part_end
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
                    if before_loc_start==-2 and before_loc_end==-2:
                        before_category_status = 2
                        before_category_mask = 0
                        before_loc_start = category_index
                        before_loc_end = category_index
                    elif before_loc_start==-1 and before_loc_end==-1:
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
                yield self.text_to_instance([sents_list, sents_anno_list, word_pos_list, part_mask_list,
                                                        before_category_status_list, before_category_mask_list,
                                                        before_loc_start_list, before_loc_end_list,
                                                        after_category_status_list, after_category_mask_list,
                                                        after_loc_start_list, after_loc_end_list])


    @overrides
    def text_to_instance(self, inputs):
        fields: Dict[str, Field] = {}

        tokens_list_field: List[TextField] = []
        sent_positions_list_field: List[TextField] = []
        position_list_field: List[TextField] = []
        participant_mask_list_field: List[SequenceLabelField] = []
        after_loc_start_list_field: List[IndexField] = []
        after_loc_end_list_field: List[IndexField] = []
        after_category_list_field: List[IndexField] = []
        after_category_mask_list_field: List[IndexField] = []

        category_list = [0, 1, 2]
        category_field: List[LabelField] = []
        for l in category_list:
            category_field.append(LabelField(str(l), "labels"))
        category_field = ListField(category_field)

        category_mask_list = [0, 1]
        category_mask_field: List[LabelField] = []
        for l in category_mask_list:
            category_mask_field.append(LabelField(str(l), "labels"))
        category_mask_field = ListField(category_mask_field)

        token_field_step0 = TextField(inputs[0][0], self._token_indexers)
        before_loc_start_field = IndexField(inputs[6][0], token_field_step0)
        before_loc_end_field = IndexField(inputs[7][0], token_field_step0)
        before_category_field = IndexField(inputs[4][0], category_field)
        before_category_mask_field = IndexField(inputs[5][0], category_mask_field)

        for i in range(len(inputs[0])):
            token_field = TextField(inputs[0][i], self._token_indexers)
            tokens_list_field.append(token_field)

            sent_position_field = TextField(inputs[1][i], self._sent_position_indexers)
            sent_positions_list_field.append(sent_position_field)

            position_field = TextField(inputs[2][i], self._token_position_indexers)
            position_list_field.append(position_field)

            participant_mask_field = SequenceLabelField(inputs[3][i], token_field, 'tags')
            participant_mask_list_field.append(participant_mask_field)

            after_loc_start_field = IndexField(inputs[10][i], token_field)
            after_loc_end_field = IndexField(inputs[11][i], token_field)

            after_loc_start_list_field.append(after_loc_start_field)
            after_loc_end_list_field.append(after_loc_end_field)

            after_category_field = IndexField(inputs[8][i], category_field)
            after_category_list_field.append(after_category_field)

            after_category_mask_field = IndexField(inputs[9][i], category_mask_field)
            after_category_mask_list_field.append(after_category_mask_field)

        fields['tokens_list'] = ListField(tokens_list_field)
        fields['positions_list'] = ListField(position_list_field)
        fields['sent_positions_list'] = ListField(sent_positions_list_field)
        fields['before_loc_start'] = before_loc_start_field
        fields['before_loc_end'] = before_loc_end_field
        fields['after_loc_start_list'] = ListField(after_loc_start_list_field)
        fields['after_loc_end_list'] = ListField(after_loc_end_list_field)
        fields['before_category'] = before_category_field
        fields['after_category_list'] = ListField(after_category_list_field)
        fields['before_category_mask'] = before_category_mask_field
        fields['after_category_mask_list'] = ListField(after_category_mask_list_field)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ProGlobalDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop("token_indexers", {}))
        token_position_indexers = TokenIndexer.dict_from_params(params.pop("token_position_indexers", {}))
        sent_position_indexers = TokenIndexer.dict_from_params(params.pop("sent_position_indexers", {}))
        return ProGlobalDatasetReader(token_indexers=token_indexers,
                                           token_position_indexers=token_position_indexers,
                                           sent_position_indexers=sent_position_indexers)


