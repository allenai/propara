# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from propara.data.prolocal_dataset_reader import ProLocalDatasetReader


class TestStateChangeDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        sc_reader = ProLocalDatasetReader()
        instances = sc_reader.read('tests/fixtures/prolocal_toy_data.tsv')
        instances = ensure_list(instances)

        # read first instance
        fields = instances[0].fields
        correct_tokens = ["Green", "plants", "absorb", "water", "from", "the", "soil"]
        read_tokens = [t.text for t in fields["tokens"].tokens]
        assert correct_tokens == read_tokens
        assert fields["entity_span"].labels == [0, 0, 0, 1, 0, 0, 0]
        assert fields["verb_span"].labels == [0, 0, 1, 0, 0, 0, 0]

        assert fields["state_change_type_labels"].label == 'MOVE'
        assert fields["state_change_tags"].labels == ['B-LOC-TO', 'I-LOC-TO', 'O', 'O', 'O', 'B-LOC-FROM', 'I-LOC-FROM']

        # read second instance
        fields = instances[1].fields
        print(fields)
        read_tokens = [t.text for t in fields["tokens"].tokens]
        assert read_tokens == ["Rocks", "in", "the", "shore", "break"]
        assert fields["entity_span"].labels == [1, 0, 0, 0, 0]
        assert fields["verb_span"].labels == [0, 0, 0, 0, 1]

        assert fields["state_change_type_labels"].label == 'DESTROY'
        assert fields["state_change_tags"].labels == ['O', 'O', 'B-LOC-FROM', 'I-LOC-FROM', 'O']
