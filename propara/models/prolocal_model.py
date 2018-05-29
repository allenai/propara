from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Linear
from overrides import overrides
import numpy

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Attention, TimeDistributed
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import F1Measure, CategoricalAccuracy
from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.nn.util import sequence_cross_entropy_with_logits


@Model.register("ProLocalModel")
class ProLocalModel(Model):
    """
    This ``Model`` takes as input a dataset read by stateChangeDatasetReader
    Input: sentence, focus entity, focus verb
    Output: state change types for the focus entity, state change tags (mainly before, after locations of focus entity)
    The basic outline of this model is to 
        1. get an embedded representation for the sentence tokens, 
        2. concatenate each token embedding with verb and entity bits,
        3. pass them through bidirectional LSTM Seq2VecEncoder
           to create a contextual sentence embedding vector,
        4. apply bilinear attention to compute attention weights over sentence tokens   
        5. apply dense layer to get most likely state_change_type among
           {Create, Destroy, Move, None}
  
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``sentence_tokens`` ``TextFields`` we get as input to the model.
    seq2seq_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output state_change_types.
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.

    Sample commandline
    ------------------
    python processes/run.py train -s /output_folder experiment_config/state_change_local.json 
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2seq_encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator) -> None:
        super(ProLocalModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder

        self.attention_layer = \
            Attention(similarity_function=BilinearSimilarity(2 * seq2seq_encoder.get_output_dim(),
                                                             seq2seq_encoder.get_output_dim()), normalize=True)

        self.num_types = self.vocab.get_vocab_size("state_change_type_labels")
        self.aggregate_feedforward = Linear(seq2seq_encoder.get_output_dim(),
                                            self.num_types)

        self.span_metric = SpanBasedF1Measure(vocab,
                                              tag_namespace="state_change_tags")  # by default "O" is ignored in metric computation
        self.num_tags = self.vocab.get_vocab_size("state_change_tags")

        self.tag_projection_layer = TimeDistributed(Linear(self.seq2seq_encoder.get_output_dim() + 2
                                                           , self.num_tags))
        self._type_accuracy = CategoricalAccuracy()

        self.type_f1_metrics = {}
        self.type_labels_vocab = self.vocab.get_index_to_token_vocabulary("state_change_type_labels")
        for type_label in self.type_labels_vocab.values():
            self.type_f1_metrics["type_" + type_label] = F1Measure(self.vocab.get_token_index(type_label, "state_change_type_labels"))

        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_span: torch.LongTensor,
                entity_span: torch.LongTensor,
                state_change_type_labels: torch.LongTensor = None,
                state_change_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        verb_span: torch.LongTensor, required.
            An integer ``SequenceLabelField`` representation of the position of the focus verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that pre-processing stage could not extract a verbal predicate.
        entity_span: torch.LongTensor, required.
            An integer ``SequenceLabelField`` representation of the position of the focus entity
            in the sentence. This should have shape (batch_size, num_tokens) 
        state_change_type_labels: torch.LongTensor, optional (default = None)
            A torch tensor representing the state change type class labels of shape ``(batch_size, 1)???
        state_change_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
            In the first implementation we focus only on state_change_types.

        Returns
        -------
        An output dictionary consisting of:
        type_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_state_change_types)`` representing
            a distribution of state change types per datapoint.
        tags_class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_state_change_types, num_tokens)`` representing
            a distribution of location tags per token in a sentence.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # Layer 1 = Word + Character embedding layer
        embedded_sentence = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        # Layer 2 = Add positional bit to encode position of focus verb and entity
        embedded_sentence_verb_entity = \
            torch.cat([embedded_sentence, verb_span.float().unsqueeze(-1), entity_span.float().unsqueeze(-1)], dim=-1)

        # Layer 3 = Contextual embedding layer using Bi-LSTM over the sentence
        contextual_embedding = self.seq2seq_encoder(embedded_sentence_verb_entity, mask)

        # Layer 4: Attention (Contextual embedding, BOW(verb span))
        verb_weight_matrix = verb_span.float() / (verb_span.float().sum(-1).unsqueeze(-1) + 1e-13)
        verb_vector = weighted_sum(contextual_embedding * verb_span.float().unsqueeze(-1), verb_weight_matrix)
        entity_weight_matrix = entity_span.float() / (entity_span.float().sum(-1).unsqueeze(-1) + 1e-13)
        entity_vector = weighted_sum(contextual_embedding * entity_span.float().unsqueeze(-1), entity_weight_matrix)
        verb_entity_vector = torch.cat([verb_vector, entity_vector], 1)
        batch_size, sequence_length, binary_feature_dim = verb_span.float().unsqueeze(-1).size()

        # attention weights for type prediction
        attention_weights_types = self.attention_layer(verb_entity_vector, contextual_embedding)
        attention_output_vector = weighted_sum(contextual_embedding, attention_weights_types)

        # contextual embedding + positional vectors for tag prediction
        context_positional_tags = torch.cat([contextual_embedding, verb_span.float().unsqueeze(-1), entity_span.float().unsqueeze(-1)], dim=-1)

        # Layer 5 = Dense softmax layer to pick one state change type per datapoint,
        # and one tag per word in the sentence
        type_logits = self.aggregate_feedforward(attention_output_vector)
        type_probs = torch.nn.functional.softmax(type_logits, dim=-1)

        tags_logits = self.tag_projection_layer(context_positional_tags)
        reshaped_log_probs = tags_logits.view(-1, self.num_tags)
        tags_class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size, sequence_length, self.num_tags])

        # Create output dictionary for the trainer
        # Compute loss and epoch metrics
        output_dict = {'type_probs': type_probs}
        if state_change_type_labels is not None:
            state_change_type_labels_loss = self._loss(type_logits, state_change_type_labels.long().view(-1))
            for type_label in self.type_labels_vocab.values():
                metric = self.type_f1_metrics["type_" + type_label]
                metric(type_probs, state_change_type_labels.squeeze(-1))

            self._type_accuracy(type_probs, state_change_type_labels.squeeze(-1))

        if state_change_tags is not None:
            state_change_tags_loss = sequence_cross_entropy_with_logits(tags_logits, state_change_tags, mask)
            self.span_metric(tags_class_probabilities, state_change_tags, mask)
            output_dict["tags_class_probabilities"] = tags_class_probabilities

        output_dict['loss'] = (state_change_type_labels_loss + state_change_tags_loss)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        predict most probable type labels
        """
        type_predictions = output_dict['type_probs']
        # batch_size, #classes=4
        type_predictions = type_predictions.cpu().data.numpy()
        argmax_indices = numpy.argmax(type_predictions, axis=-1)
        type_labels = [self.vocab.get_token_from_index(x, namespace="state_change_type_labels")
                       for x in argmax_indices]
        output_dict['predicted_types'] = type_labels

        # predict most probable tag labels
        tag_predictions = output_dict['tags_class_probabilities']
        tag_predictions = tag_predictions.cpu().data.numpy()

        if tag_predictions.ndim == 3:
            predictions_list = [tag_predictions[i] for i in range(tag_predictions.shape[0])]
        else:
            predictions_list = [tag_predictions]
        all_tags: List[List[str]] = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace="state_change_tags")
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict['predicted_tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.span_metric.get_metric(reset=reset)

        type_accuracy = self._type_accuracy.get_metric(reset)
        metric_dict['type_accuracy'] = type_accuracy

        for name, metric in self.type_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]

        metric_dict['combined_metric'] = type_accuracy * metric_dict['f1-measure-overall']

        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ProLocalModel':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        seq2seq_encoder_params = params.pop("seq2seq_encoder")
        seq2seq_encoder = Seq2SeqEncoder.from_params(seq2seq_encoder_params)

        initializer = InitializerApplicator.from_params(params.pop("initializer", []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   seq2seq_encoder=seq2seq_encoder,
                   initializer=initializer)
