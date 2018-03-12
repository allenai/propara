from allennlp.common.testing import AllenNlpTestCase

from propara.data.proglobal_dataset_reader import ProGlobalDatasetReader

class TestDataReader(AllenNlpTestCase):

    class TestStateChangeDatasetReader(AllenNlpTestCase):
        def test_read_from_file(self):
            sc_reader = ProGlobalDatasetReader()
            dataset = sc_reader.read('tests/fixtures/proglobal_toy_data.tsv')
            instances = dataset.instances

            # read first instance
            fields = instances[0].fields
            assert len(fields["tokens"].tokens) == len(fields["positions"].tokens)
            assert len(fields["tokens"].tokens)==len(fields["sent_positions"].tokens)
            part_mask_field = fields["participant_mask_list"].field_list[0]
            labels = part_mask_field.labels
            sequence_field = part_mask_field.sequence_field

            assert len(labels)==sequence_field.sequence_length()