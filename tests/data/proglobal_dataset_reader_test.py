from propara.data.proglobal_dataset_reader import ProGlobalDatasetReader

from allennlp.common.testing import AllenNlpTestCase


class TestDataReader(AllenNlpTestCase):
    def test_read_from_file(self):
        sc_reader = ProGlobalDatasetReader()
        dataset = sc_reader.read('tests/fixtures/proglobal_toy_data.tsv')
        instances = dataset
        assert len(instances) == 20

        # read first instance
        fields = instances[0].fields
        assert fields["tokens_list"].sequence_length() == fields["positions_list"].sequence_length()
        tokens_list_fields = fields["tokens_list"].field_list
        field0 = tokens_list_fields[0]
        field0_tokens = [t.text for t in field0.tokens[0:10]]
        correct_field0_tokens = ["when", "water", "freeze", "it", "become", "10", "%", "bigger", ",", "or"]
        assert field0_tokens == correct_field0_tokens

        before_loc_start_field = fields["before_loc_start"].sequence_index
        before_loc_end_field = fields["before_loc_end"].sequence_index
        assert before_loc_start_field == 0
        assert before_loc_end_field == 0

        after_loc_start_fields = fields["after_loc_start_list"].field_list
        after_loc_end_fields = fields["after_loc_end_list"].field_list
        after_loc_start_fields0 = after_loc_start_fields[0].sequence_index
        after_loc_end_fields0 = after_loc_end_fields[0].sequence_index
        assert after_loc_start_fields0 == 0
        assert after_loc_end_fields0 == 0

        before_category = fields["before_category"].sequence_index
        assert before_category == 1

        after_category_fields = fields["after_category_list"].field_list
        after_category_fields0 = after_category_fields[0].sequence_index
        assert after_category_fields0 == 1

        before_category_mask = fields["before_category_mask"].sequence_index
        assert before_category_mask == 0

        after_category_mask_fields = fields["after_category_mask_list"].field_list
        after_category_mask_fields0 = after_category_mask_fields[0].sequence_index
        assert after_category_mask_fields0 == 0




