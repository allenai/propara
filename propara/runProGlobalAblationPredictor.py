import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from allennlp.commands import main
from propara.models.proglobal_model_ablation import ProGlobalAblation
from propara.data.proglobal_dataset_reader import ProGlobalDatasetReader
from propara.service.predictors.proglobal_prediction import ProGlobalPredictor

if __name__ == "__main__":
    predictor_overrides = {'ProGlobalAblation': 'ProGlobalPrediction'}
    main(prog="python -m allennlp.run")#, predictor_overrides=predictor_overrides)
