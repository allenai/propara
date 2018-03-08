# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from propara.data.prolocal_dataset_reader import ProLocalDatasetReader
from propara.models.prolocal_model import ProLocalModel
from propara.service.predictors.prolocal_prediction import ProLocalPredictor

class TestStateChangePredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {"instance": "test++++460++++1++++living things have carbon in them++++carbon++++living####things####have####carbon####in####them++++0,0,0,0,0,0++++0,0,0,1,0,0++++NONE++++O,O,O,O,O,O"}

        archive = load_archive('tests/fixtures/prolocal_toy_model.tar.gz')
        predictor = Predictor.from_archive(archive, 'prolocal-prediction')
        result = predictor.predict_json(inputs)
        assert(len(result) > 0)
