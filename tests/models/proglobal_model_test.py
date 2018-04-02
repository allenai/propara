from allennlp.common.testing import ModelTestCase
from flaky import flaky
from propara.data.proglobal_dataset_reader import ProGlobalDatasetReader
from propara.models.proglobal_model import ProGlobal

class ProGlobalModelTest(ModelTestCase):
    def setUp(self):
        super(ProGlobalModelTest, self).setUp()
        self.set_up_model('tests/fixtures/proglobal_params.json', 'tests/fixtures/proglobal_toy_data.tsv')

    def test_proglobal_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def ensure_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
