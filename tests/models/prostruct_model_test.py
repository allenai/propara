# pylint: disable=no-self-use,invalid-name
import os

from allennlp.common.testing import ModelTestCase
from flaky import flaky
from propara.data.propara_dataset_reader import ProParaDatasetReader

class ProStructModelTest(ModelTestCase):
    def setUp(self):
        super(ProStructModelTest, self).setUp()
        # Make testing with elmo optional because:
        # 1. It takes too long for a test case to train with elmo.
        # 2. Elmo model files are too large for Git, so a local path
        #    must be setup in test config json.
        self.set_up_model(
                'tests/fixtures/prostruct/prostruct_model_test.json',
                'tests/fixtures/prostruct/toy_data_prostruct_model'
        )

    def test_state_change_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def ensure_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
