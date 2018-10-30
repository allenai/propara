from overrides import overrides
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from propara.models.prolocal_model import ProLocalModel
from propara.data.prolocal_dataset_reader import ProLocalDatasetReader

@Predictor.register('prolocal-prediction')
class ProLocalPredictor(Predictor):
    """
    Wrapper for the :class:`propara.models.StateChangeModelCombinedLossBothAttention` model.
    """
    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
       instance_text = inputs["instance"]
       parts: List[str] = instance_text.split("++++")

       print(len(parts))
       print(instance_text)

       paraid = parts[1]
       sentenceid = parts[2]
       sentence = parts[3]
       entity = parts[4]

       sentence_tokens = parts[5].split("####")
       verb_span = parts[6].split(",")
       verb_vector = [int(i) for i in verb_span]
       entity_span = parts[7].split(",")
       entity_vector = [int(i) for i in entity_span]

       state_change_types = parts[8]
       state_change_tags = parts[9].split(",")

       instance = self._dataset_reader.text_to_instance(sentence_tokens=sentence_tokens,
                                                        verb_vector=verb_vector,
                                                        entity_vector=entity_vector,
                                                        state_change_types=state_change_types,
                                                        state_change_tags=state_change_tags)
       outputs = self._model.forward_on_instance(instance=instance)

       predictions = {}
       predictions["paraid"] = paraid
       predictions["sentenceid"] = sentenceid
       predictions["sentence"] = sentence
       predictions["sentence_tokens"] = sentence_tokens
       predictions["entity"] = entity

       predictions["predicted_types"] = outputs["predicted_types"]
       predictions["predicted_tags"] = outputs["predicted_tags"]

       predictions["gold_types"] = state_change_types
       predictions["gold_tags"] = state_change_tags

       return predictions
