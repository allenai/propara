# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from propara.service.predictors.prostruct_prediction import ProStructPredictor


class TestProParaPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {"para_id": "4",
                  "sentence_texts": ["Plants die.",
                                     "They are buried in sediment.",
                                     "Bacteria is buried in the sediment.",
                                     "Large amounts of sediment gradually pile on top of the original sediment.",
                                     "Pressure builds up.",
                                     "Heat increases.",
                                     "The chemical structure of the buried sediment and plants changes.",
                                     "The sediment and plants are at least one mile underground.",
                                     "The buried area is extremely hot.",
                                     "More chemical changes happen eand the buried material becomes oil."
                                     ],
                  "participants": ["plants",
                                   "bacteria",
                                   "sediment",
                                   "oil"],
                  "states": [
                      ["?", "?", "sediment", "sediment", "sediment", "sediment", "sediment", "sediment", "one mile underground", "one mile underground", "-"],
                      ["?", "?", "?", "sediment", "sediment", "sediment", "sediment", "sediment", "sediment", "sediment", "-"],
                      ["?", "?", "?", "?", "?", "?", "?", "?", "underground", "underground", "underground"],
                      ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "underground"]]}


        archive = load_archive('tests/fixtures/prostruct/prostruct_toy_model.tar.gz')
        predictor = Predictor.from_archive(archive, 'prostruct_prediction')
        result = predictor.predict_json(inputs)
        assert(result['para_id'] == '4')
        assert(result["sentence_texts"] == ["Plants die.",
                                            "They are buried in sediment.",
                                            "Bacteria is buried in the sediment.",
                                            "Large amounts of sediment gradually pile on top of the original sediment.",
                                            "Pressure builds up.",
                                            "Heat increases.",
                                            "The chemical structure of the buried sediment and plants changes.",
                                            "The sediment and plants are at least one mile underground.",
                                            "The buried area is extremely hot.",
                                            "More chemical changes happen eand the buried material becomes oil."
                                            ])
        assert(result['participants'] == ["plants",
                                          "bacteria",
                                          "sediment",
                                          "oil"])
        # This changes with a new model (but some label must be predicted).
        print(f"result['top1_labels']: {result['top1_labels']}")
        assert(len(result['top1_labels']) > 1)
