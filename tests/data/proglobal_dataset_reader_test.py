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