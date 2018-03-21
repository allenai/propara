# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import ModelTestCase
from flaky import flaky
from propara.data.prolocal_dataset_reader import ProLocalDatasetReader
from propara.models.prolocal_model import ProLocalModel

class ProLocalModelTest(ModelTestCase):
    def setUp(self):
        super(ProLocalModelTest, self).setUp()
        self.set_up_model('tests/fixtures/prolocal_params.json', 'tests/fixtures/prolocal_toy_data.tsv')

    def test_state_change_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def ensure_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
