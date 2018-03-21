from typing import Dict, List, Optional
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.tokenizers.token import Token
from allennlp.data.fields.field import Field  # pylint: disable=unused-import

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("ProLocalDatasetReader")
class ProLocalDatasetReader(DatasetReader):
    """
    Reads a file from ProPara state change dataset.  This data is formatted as TSV, one instance per line.
    Format: "tokenized_sentence \t entity_span \t verb_span \t state_change_types \t state_change_tags"
    tokenized_sentence: tokens are separated by '####'
    entity_span: boolean sequence of length = sentence_length, separated by ',', indicating the entity span 
    verb_span: boolean sequence of length = sentence_length, separated by ',', indicating the verb span 

    state_change_types: string label applicable to this datapoint
    state_change_tags: state change tag string per token, separated by ','

    We convert these columns into fields named 
    "sentence_tokens", "entity_span", "verb_span", 
    "state_change_types", "state_change_tag".

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []
        with open(file_path, 'r') as state_change_file:
            logger.info("Reading state change instances from TSV dataset at: %s", file_path)
            for line in tqdm.tqdm(state_change_file):
                parts: List[str] = line.split()
                # parse input
                sentence_tokens = parts[0].split("####")
                verb_span = parts[1].split(",")
                verb_vector = [int(i) for i in verb_span]
                entity_span = parts[2].split(",")
                entity_vector = [int(i) for i in entity_span]

                # parse labels
                state_change_types = parts[3]
                state_change_tags = parts[4].split(",")

                # create instance
                yield self.text_to_instance(sentence_tokens=sentence_tokens,
                                                       verb_vector=verb_vector,
                                                       entity_vector=entity_vector,
                                                       state_change_types=state_change_types,
                                                       state_change_tags=state_change_tags)


    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence_tokens: List[str],
                         verb_vector: List[int],
                         entity_vector: List[int],
                         state_change_types: Optional[List[str]] = None,
                         state_change_tags: Optional[List[str]] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        # encode inputs
        token_field = TextField([Token(word) for word in sentence_tokens], self._token_indexers)
        fields['tokens'] = token_field
        fields['verb_span'] = SequenceLabelField(verb_vector, token_field, 'indicator_tags')
        fields['entity_span'] = SequenceLabelField(entity_vector, token_field, 'indicator_tags')

        # encode outputs
        if state_change_types:
            fields['state_change_type_labels'] = LabelField(state_change_types, 'state_change_type_labels')
        if state_change_tags:
            fields['state_change_tags'] = SequenceLabelField(state_change_tags, token_field, 'state_change_tags')

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ProLocalDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop("token_indexer", {}))
        params.assert_empty(cls.__name__)
        return ProLocalDatasetReader(token_indexers=token_indexers)
