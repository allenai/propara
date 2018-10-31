from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Linear
from overrides import overrides
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import nll_loss

from allennlp.common import Params

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Attention, TimeDistributed
from allennlp.nn import util, InitializerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import F1Measure, CategoricalAccuracy, BooleanAccuracy
from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity
from allennlp.training.metrics import SpanBasedF1Measure, SquadEmAndF1
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.decoding import BeamSearch
from propara.commonsense.background_knowledge.kb3_lexical import KBLexical
from propara.trainer_decoder.maximum_marginal_likelihood import MaximumMarginalLikelihood

from propara.data.propara_dataset_reader import Action
from propara.trainer_decoder.action_scorer import ActionScorerDummy, KBBasedActionScorer
from propara.trainer_decoder.propara_trainer_decoder_helper import DecoderTrainerHelper
from propara.trainer_decoder.propara_decoder_step import ProParaDecoderStep
from propara.trainer_decoder.valid_action_generator import DummyConstrainedStepper, CommonsenseBasedActionGenerator


@Model.register("ProStructModel")
class ProStructModel(Model):
    """
    This ``Model`` takes as input a dataset read by ProParaDatasetReader
    Input: paragraph, sentences, participants, verbs,
           labels: action-type, before_locations, after_locations

    Output: state change types for each participant at every step
    The basic outline of this model is to
     For each sentence
        For each participant
            1. get an embedded representation for the sentence tokens,
            2. concatenate each token embedding with verb and participant bits,
            3. pass them through bidirectional LSTM Seq2VecEncoder
               to create a contextual sentence embedding vector,
            4. apply dense layer to get most likely action-type
               {Create, Destroy, Move, None}

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``sentence_tokens`` ``TextFields`` we get as input to the model.
    use_attention : ``bool``
        If ``True``, the model will apply the seq2seq_encoder
    seq2seq_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output state_change_types.
    seq2vec_encoder : ``Seq2VecEncoder``
    span_end_encoder_after : ``Seq2SeqEncoder``
    use_decoder_trainer: bool
    decoder_beam_search: ``BeamSearch``
    kb_configs: ``dict``
    other_configs: ``dict``
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.

    Sample commandline
    ------------------
    python propara/run.py train -s /output_folder experiment_config/propara_local.json
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 use_attention: bool,
                 seq2seq_encoder: Seq2SeqEncoder,
                 seq2vec_encoder: Seq2VecEncoder,
                 span_end_encoder_after: Seq2SeqEncoder,
                 use_decoder_trainer: bool,
                 decoder_beam_search: BeamSearch,
                 kb_configs: dict,
                 other_configs: dict,
                 initializer: InitializerApplicator) -> None:
        super(ProStructModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.num_actions = len(Action)  # number of actions is hardcoded here.
        # They are defined in Action enum in propara_dataset_reader.py
        self.other_configs = other_configs

        # kb_coefficient * kb_score + (1-kb_coefficient) * model_score
        self.kb_coefficient = torch.nn.Parameter(torch.ones(1).mul(kb_configs.get('kb_coefficient', 0.5)))

        self.use_attention = use_attention
        self.use_decoder_trainer = use_decoder_trainer
        if self.use_attention:
            self.seq2seq_encoder = seq2seq_encoder
            self.time_distributed_seq2seq_encoder = TimeDistributed(TimeDistributed(self.seq2seq_encoder))
            self.time_distributed_attention_layer = \
                TimeDistributed(TimeDistributed(
                    Attention(similarity_function=BilinearSimilarity(2 * seq2seq_encoder.get_output_dim(),
                                                                     seq2seq_encoder.get_output_dim()),
                              normalize=True)))
            self.aggregate_feedforward = Linear(seq2seq_encoder.get_output_dim(),
                                                self.num_actions)
        else:
            self.seq2vec_encoder = seq2vec_encoder
            self.time_distributed_seq2vec_encoder = TimeDistributed(TimeDistributed(self.seq2vec_encoder))
            self.aggregate_feedforward = Linear(seq2vec_encoder.get_output_dim(),
                                                self.num_actions)

        self.span_end_encoder_after = span_end_encoder_after
        # per step per participant
        self.time_distributed_encoder_span_end_after = TimeDistributed(TimeDistributed(self.span_end_encoder_after))

        # Fixme: dimensions

        self._span_start_predictor_after = TimeDistributed(TimeDistributed(torch.nn.Linear(2 + 2*seq2seq_encoder.get_output_dim(), 1)))

        self._span_end_predictor_after = TimeDistributed(TimeDistributed(torch.nn.Linear(span_end_encoder_after.get_output_dim(), 1)))

        self._type_accuracy = BooleanAccuracy()
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Fixme: This is less robust. If the masking value

        # Fixme: add a metric for location span strings
        self.span_metric = SquadEmAndF1()

        if self.use_decoder_trainer:
            self.decoder_trainer = MaximumMarginalLikelihood()
            if kb_configs['kb_to_use'] == 'lexicalkb':
                kb = KBLexical(
                    lexical_kb_path=kb_configs['lexical_kb_path'],
                    fullgrid_prompts_load_path=kb_configs['fullgrid_prompts_load_path']
                )

            # Makeshift arrangement to get number of participants in tiny.tsv .
            self.commonsense_based_action_generator = CommonsenseBasedActionGenerator(self.num_actions)
            self.rules_activated = [int(rule_val.strip()) > 0
                                    for rule_val in self.other_configs.get('constraint_rules_to_turn_on', '0,0,0,1')
                                                        .split(",")]
            self.rule_2_fraction_participants = self.other_configs.get('rule_2_fraction_participants', 0.5)
            self.rule_3_fraction_steps = self.other_configs.get('rule_3_fraction_steps', 0.5)

            self.commonsense_based_action_generator.set_rules_used(self.rules_activated,
                                                                   self.rule_2_fraction_participants,
                                                                   self.rule_3_fraction_steps)
            # [self.rules_activated[0],  # C/D/C/D cannot happen
            #  self.rules_activated[1],  # > 1/2 partic
            #  self.rules_activated[2],  # > 1/2 steps cannot change
            #  self.rules_activated[3]  # until mentioned
            #  ])
            self.decoder_step = ProParaDecoderStep(KBBasedActionScorer(kb=kb, kb_coefficient=self.kb_coefficient),
                                                   valid_action_generator=self.commonsense_based_action_generator)

        self.beam_search = decoder_beam_search
        initializer(self)


    def forward(self,  # type: ignore
                para_id: int,
                participant_strings: List[str],
                paragraph: Dict[str, torch.LongTensor],
                sentences: Dict[str, torch.LongTensor],
                paragraph_sentence_indicators: torch.IntTensor,
                participants: Dict[str, torch.LongTensor],
                participant_indicators: torch.IntTensor,
                paragraph_participant_indicators: torch.IntTensor,
                verbs: torch.IntTensor,
                paragraph_verbs: torch.IntTensor,
                actions: torch.IntTensor = None,
                before_locations: torch.IntTensor = None,
                after_locations: torch.IntTensor = None,
                filename: List[str] = [],
                score: List[float] = 1.0  # instance_score
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        para_id: int
            The id of the paragraph
        participant_strings: List[str]
            The participants in the paragraph
        paragraph: Dict[str, torch.LongTensor]
            The token indices for the paragraph
        sentences: Dict[str, torch.LongTensor]
            The token indices batched by sentence.
        paragraph_sentence_indicators: torch.LongTensor
            Indicates before / inside / after for each sentence
        participants: Dict[str, torch.LongTensor]
            The token indices for the participant names
        participant_indicators: torch.IntTensor
            Indicates each participant in each sentence
        paragraph_participant_indicators: torch.IntTensor
            Indicates each participant in the paragraph
        verbs: torch.IntTensor
            Indicates the positions of verbs in the sentences
        paragraph_verbs: torch.IntTensor
            Indicates the positions of verbs in the paragraph
        actions: torch.IntTensor, optional (default = None)
            Indicates the actions taken per participant
            per sentence.
        before_locations: torch.IntTensor, optional (default = None)
            Indicates the span for the before location
            per participant per sentence
        after_locations: torch.IntTensor, optional (default = None)
            Indicates the span for the after location
            per participant per sentence
        filename: List[str], optional (default = '')
            The files from which the instances were read
        score: List[float], optional (default = 1.0)
            The score for each instance

        Returns
        -------
        An output dictionary consisting of:
        action_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_sentences, num_participants, num_action_types)`` representing
            a distribution of state change types per sentence, participant in each datapoint (paragraph).
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        self.filename = filename
        self.instance_score = score

        # original shape (batch_size, num_participants, num_sentences, sentence_length)
        participant_indicators = participant_indicators.transpose(1, 2)
        # new shape (batch_size, num_sentences, num_participants, sentence_length)

        batch_size, num_sentences, num_participants, sentence_length = participant_indicators.size()

        # (batch_size, num_sentences, sentence_length, embedding_size)
        embedded_sentences = self.text_field_embedder(sentences)
        # (batch_size, num_participants, description_length, embedding_size)
        embedded_participants = self.text_field_embedder(participants)

        batch_size, num_sentences, sentence_length, embedding_size = embedded_sentences.size()
        self.num_sentences = num_sentences

        # ===========================================================================================================
        # Layer 1: For each sentence, participant pair: create a Glove embedding for each token
        # (batch_size, num_sentences, num_participants, sentence_length, embedding_size)
        embedded_sentence_participant_pairs = embedded_sentences.unsqueeze(2).expand(batch_size, num_sentences, \
                                                                                     num_participants, sentence_length,
                                                                                     embedding_size)

        # (batch_size, num_sentences, sentence_length) -> (batch_size, num_sentences, num_participants, sentence_length)
        mask = get_text_field_mask(sentences, num_wrapping_dims=1). \
            unsqueeze(2).expand(batch_size, num_sentences, num_participants, sentence_length).float()

        # (batch_size, num_participants, num_sentences * sentence_length)
        participant_view = participant_indicators.transpose(1, 2). \
            view(batch_size, num_participants, num_sentences * sentence_length)

        # participant_mask is used to mask out invalid sentence, participant pairs
        # (batch_size, num_sentences, num_participants, sentence_length)
        sent_participant_pair_mask = (participant_view.sum(dim=2) > 0). \
            unsqueeze(-1).expand(batch_size, num_participants, num_sentences). \
            unsqueeze(-1).expand(batch_size, num_participants, num_sentences, sentence_length). \
            transpose(1, 2).float()

        # whether the sentence is masked or not (sent does not exist in paragraph).
        # this is either (batch_size, num_sentences, num_participants)
        # or if only one participant (batch_size, num_sentences)
        # TODO(joelgrus) why is there a squeeze here
        sentence_mask = (mask.sum(3) > 0).squeeze(-1).float()

        # (batch_size, num_sentences, num_participants, sentence_length)
        mask = mask * sent_participant_pair_mask

        # (batch_size, num_participants, num_sentences * sentence_length)
        # -> (batch_size, num_participants)
        # -> (batch_size, num_participants, num_sentences)
        # -> (batch_size, num_sentences, num_participants)
        participant_mask = (participant_view.sum(dim=2) > 0). \
            unsqueeze(-1).expand(batch_size, num_participants, num_sentences). \
            transpose(1, 2).float()

        # Example: 0.0 where action is -1 (padded)
        # action:  [[[1, 0, 1], [3, 2, 3]], [[0, -1, -1], [-1, -1, -1]]]
        # action_mask:  [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        # (batch_size, num_sentences, num_participants)
        action_mask = participant_mask * sentence_mask

        # (batch_size, num_sentences, num_participants, sentence_length)
        verb_indicators = verbs.unsqueeze(2).expand(batch_size, num_sentences, num_participants,
                                                    sentence_length).float()

        # ===========================================================================================================
        # Layer 2: Concatenate sentence embedding with verb and participant indicator bits
        # espp: (batch_size, num_sentences, num_participants, sentence_length, embedding_size)
        # vi:   (batch_size, num_sentences, num_participants, sentence_length)
        # pi:   (batch_size, num_sentences, num_participants, sentence_length)
        #
        # result: (batch_size, num_sentences, num_participants, sentence_length, embedding_size + 2)
        embedded_sentence_verb_entity = \
            torch.cat([embedded_sentence_participant_pairs, verb_indicators.unsqueeze(-1).float(),
                       participant_indicators.unsqueeze(-1).float()], dim=-1)

        # ===========================================================================================================
        # Layer 3 = Contextual embedding layer using Bi-LSTM over the sentence

        if self.use_attention:
            # (batch_size, num_sentences, num_participants, sentence_length, )
            # contextual_seq_embedding: batch_size * num_sentences *
        #                                num_participants * sentence_length * (2*seq2seq_output_size)
            contextual_seq_embedding = self.time_distributed_seq2seq_encoder(embedded_sentence_verb_entity, mask)

            # Layer 3.5: Attention (Contextual embedding, BOW(verb span))
            verb_weight_matrix = verb_indicators.float() / (verb_indicators.float().sum(-1).unsqueeze(-1) + 1e-13)
            # (batch_size, num_sentences, num_participants, embedding_size)
            verb_vector = weighted_sum(contextual_seq_embedding * verb_indicators.float().unsqueeze(-1),
                                       verb_weight_matrix)

            # (batch_size, num_sentences, num_participants, sentence_length)
            participant_weight_matrix = participant_indicators.float() / (
                participant_indicators.float().sum(-1).unsqueeze(-1) + 1e-13)

            # (batch_size, num_sentences, num_participants, embedding_size)
            participant_vector = weighted_sum(contextual_seq_embedding * participant_indicators.float().unsqueeze(-1),
                                              participant_weight_matrix)

            # (batch_size, num_sentences, num_participants, 2 * embedding_size)
            verb_participant_vector = torch.cat([verb_vector, participant_vector], -1)
            batch_size, num_sentences, num_participants, sentence_length, verb_ind_size = verb_indicators.float().unsqueeze(
                -1).size()

            # attention weights for type prediction
            # (batch_size, num_sentences, num_participants)
            attention_weights_actions = self.time_distributed_attention_layer(verb_participant_vector,
                                                                              contextual_seq_embedding, mask)
            contextual_vec_embedding = weighted_sum(contextual_seq_embedding, attention_weights_actions)

        else:
            # batch_size * num_sentences * num_participants * sentence_length * embedding_size
            contextual_vec_embedding = self.time_distributed_seq2vec_encoder(embedded_sentence_verb_entity, mask)

        # (batch_size, num_participants, num_sentences, 1) -> (batch_size, nnum_sentences, num_participants, 1)
        if actions is not None:
            actions = actions.transpose(1, 2)

        # # ===========================================================================================================
        # # Layer 4 = Aggregate FeedForward to choose an action label per sentence, participant pair
        # (batch_size, num_sentences, num_participants, num_actions)
        action_logits = self.aggregate_feedforward(contextual_vec_embedding)

        action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
        # (batch_size * num_sentences * num_participants, num_actions)
        action_probs_decode = action_probs.view((batch_size * num_sentences * num_participants), self.num_actions)

        output_dict = {}
        if self.use_decoder_trainer:
            # (batch_size, num_participants, description_length, embedding_size)
            participants_list = embedded_participants.data.cpu().numpy()

            output_dict.update(DecoderTrainerHelper.pass_on_info_to_decoder_trainer(
                selfie=self,
                para_id_list=para_id,
                actions=actions,
                target_mask=action_mask,
                participants_list=participants_list,
                participant_strings=participant_strings,
                participant_indicators=participant_indicators.transpose(1, 2),
                logit_tensor=action_logits
            ))

            # Compute type_accuracy based on best_final_states and actions
            best_decoded_state = output_dict['best_final_states'][0][0][0]
            best_decoded_action_seq = []
            if best_decoded_state.action_history:
                for cur_step_action in best_decoded_state.action_history[0]:
                    step_predictions = []
                    for step_action in list(cur_step_action):
                        step_predictions.append(step_action)
                    best_decoded_action_seq.append(step_predictions)
                best_decoded_tensor = torch.LongTensor(best_decoded_action_seq).unsqueeze(0)

                if actions is not None:
                    flattened_gold = actions.long().contiguous().view(-1)
                    self._type_accuracy(best_decoded_tensor.long().contiguous().view(-1), flattened_gold)
            output_dict['best_decoded_action_seq'] = [best_decoded_action_seq]
        else:
            # Create output dictionary for the trainer
            # Compute loss and epoch metrics
            output_dict["action_probs"] = action_probs
            output_dict["action_probs_decode"] = action_probs_decode

            action_loss = 0.0
            location_loss = 0.0
            if actions is not None:
                # (batch_size * num_sentences * num_participants, num_actions)
                flattened_predictions = action_logits.view((batch_size * num_sentences * num_participants),
                                                           self.num_actions)
                # Flattened_gold: contains the gold action index (Action enum in propara_dataset_reader)
                # Note: tensor is not a single block of memory, but a block with holes.
                # view can be only used with contiguous tensors, so if you need to use it here, just call .contiguous() before.
                # (batch_size * num_sentences * num_participants)
                flattened_gold = actions.long().contiguous().view(-1)
                action_loss = self._loss(flattened_predictions, flattened_gold)
                flattened_probs = action_probs.view((batch_size * num_sentences * num_participants), self.num_actions)
                evaluation_mask = (flattened_gold != -1)

                self._type_accuracy(flattened_probs, flattened_gold, mask=evaluation_mask)
                output_dict["loss"] = action_loss

        best_span_after, span_start_logits_after, span_end_logits_after = \
            self.compute_location_spans(contextual_seq_embedding=contextual_seq_embedding,
                                        embedded_sentence_verb_entity=embedded_sentence_verb_entity,
                                        mask=mask)
        output_dict["location_span_after"] =[best_span_after]

        not_in_test = (self.training or 'test' not in self.filename)

        if not_in_test and (before_locations is not None and after_locations is not None):
            after_locations = after_locations.transpose(1, 2)

            (bs, ns, np, sl) = span_start_logits_after.size()
            #print("after_locations[:,:,:,[0]]:", after_locations[:,:,:,[0]])

            location_mask = (after_locations[:, :, :, 0] >= 0).float().unsqueeze(-1).expand(bs, ns, np, sl)

            #print("location_mask:", location_mask)

            start_after_log_predicted = util.masked_log_softmax(span_start_logits_after, location_mask)
            start_after_log_predicted_transpose = start_after_log_predicted.transpose(2, 3).transpose(1,2)
            start_after_gold = torch.clamp(after_locations[:, :, :, [0]].squeeze(-1), min=-1)
            #print("start_after_log_predicted_transpose: ", start_after_log_predicted_transpose)
            #print("start_after_gold: ", start_after_gold)
            location_loss = nll_loss(input=start_after_log_predicted_transpose, target=start_after_gold, ignore_index=-1)

            end_after_log_predicted = util.masked_log_softmax(span_end_logits_after, location_mask)
            end_after_log_predicted_transpose = end_after_log_predicted.transpose(2, 3).transpose(1, 2)
            end_after_gold = torch.clamp(after_locations[:, :, :, [1]].squeeze(-1), min=-1)
            #print("end_after_log_predicted_transpose: ", end_after_log_predicted_transpose)
            #print("end_after_gold: ", end_after_gold)
            location_loss += nll_loss(input=end_after_log_predicted_transpose, target=end_after_gold,
                                     ignore_index=-1)
            output_dict["loss"] += location_loss
            # output_dict = {"loss" : 0.0}

        output_dict['action_probs_decode'] = action_probs_decode
        output_dict['action_logits'] = action_logits
        return output_dict

    def compute_location_spans(self, contextual_seq_embedding, embedded_sentence_verb_entity, mask):
        # # ===============================================================test============================================
        # # Layer 5: Span prediction for before and after location
        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        batch_size, num_sentences, num_participants, sentence_length, encoder_dim = contextual_seq_embedding.shape
        #print("contextual_seq_embedding: ", contextual_seq_embedding.shape)
        # size(span_start_input_after): batch_size * num_sentences *
        #                                num_participants * sentence_length * (embedding_size+2+2*seq2seq_output_size)
        span_start_input_after = torch.cat([embedded_sentence_verb_entity, contextual_seq_embedding], dim=-1)

        #print("span_start_input_after: ", span_start_input_after.shape)
        # Shape: (bs, ns , np, sl)
        span_start_logits_after = self._span_start_predictor_after(span_start_input_after).squeeze(-1)
        #print("span_start_logits_after: ", span_start_logits_after.shape)

        # Shape: (bs, ns , np, sl)
        span_start_probs_after = util.masked_softmax(span_start_logits_after, mask)
        #print("span_start_probs_after: ", span_start_probs_after.shape)

        # span_start_representation_after: (bs, ns , np, encoder_dim)
        span_start_representation_after = util.weighted_sum(contextual_seq_embedding, span_start_probs_after)
        #print("span_start_representation_after: ", span_start_representation_after.shape)

        # span_tiled_start_representation_after: (bs, ns , np, sl, 2*seq2seq_output_size)
        span_tiled_start_representation_after = span_start_representation_after.unsqueeze(3).expand(batch_size,
                                                                                                    num_sentences,
                                                                                                    num_participants,
                                                                                                    sentence_length,
                                                                                                    encoder_dim)
        #print("span_tiled_start_representation_after: ", span_tiled_start_representation_after.shape)

        # Shape: (batch_size, passage_length, (embedding+2  + encoder_dim + encoder_dim + encoder_dim))
        span_end_representation_after = torch.cat([embedded_sentence_verb_entity,
                                                   contextual_seq_embedding,
                                                   span_tiled_start_representation_after,
                                                   contextual_seq_embedding * span_tiled_start_representation_after],
                                                  dim=-1)
        #print("span_end_representation_after: ", span_end_representation_after.shape)

        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end_after = self.time_distributed_encoder_span_end_after(span_end_representation_after, mask)
        #print("encoded_span_end_after: ", encoded_span_end_after.shape)

        span_end_logits_after = self._span_end_predictor_after(encoded_span_end_after).squeeze(-1)
        #print("span_end_logits_after: ", span_end_logits_after.shape)

        span_end_probs_after = util.masked_softmax(span_end_logits_after, mask)
        #print("span_end_probs_after: ", span_end_probs_after.shape)

        span_start_logits_after = util.replace_masked_values(span_start_logits_after, mask, -1e7)
        span_end_logits_after = util.replace_masked_values(span_end_logits_after, mask, -1e7)

        # Fixme: we should condition this on predicted_action so that we can output '-' when needed
        # Fixme: also add a functionality to be able to output '?': we can use span_start_probs_after, span_end_probs_after
        best_span_after = self.get_best_span(span_start_logits_after, span_end_logits_after)
        #print("best_span_after: ", best_span_after)
        return best_span_after, span_start_logits_after, span_end_logits_after


    @staticmethod
    def get_best_span(span_start_logits: Variable, span_end_logits: Variable) -> Variable:
        if span_start_logits.dim() != 4 or span_end_logits.dim() != 4:
            raise ValueError("Input shapes must be (batch_size, num_sentences, num_participants, sentence_length)")
        batch_size, num_sentences, num_participants, sentence_length = span_start_logits.size()
        max_span_log_prob = -1e20*np.ones([batch_size, num_sentences, num_participants, 1], dtype=float)
        span_start_argmax = np.zeros([batch_size, num_sentences, num_participants, 1], dtype=int)
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, num_sentences, num_participants, 2).fill_(0)).int()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for s in range(num_sentences):
                for p in range(num_participants):
                    for j in range(sentence_length):
                        # for each b, s, p
                        # we are evaluating spans ending on j
                        # span_start_argmax[b,s,p]: best start_span from 0 to j (including)

                        val1 = span_start_logits[b,s,p, span_start_argmax[b,s,p]]
                        if val1 < span_start_logits[b,s,p, j]:
                            span_start_argmax[b,s,p] = j
                            val1 = span_start_logits[b,s,p, j]

                        val2 = span_end_logits[b,s,p, j]

                        if val1 + val2 > max_span_log_prob[b,s,p]:
                            best_word_span[b, s, p, 0] = int(span_start_argmax[b,s,p,0])
                            best_word_span[b, s, p, 1] = j
                            max_span_log_prob[b,s,p] = val1 + val2
        #print("best_word_span: ", best_word_span)
        return best_word_span

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        predict most probable action labels
        """
        # Fixme: Check if padded instances are being ignored. Output only the valid predictions.

        # print("In model.decode() function: ")
        # print(output_dict)
        action_probs_decode = output_dict['action_probs_decode']
        # batch_size, #classes=4
        action_probs_decode = action_probs_decode.cpu().data.numpy()
        if action_probs_decode.ndim == 3:
            predictions_list = [action_probs_decode[i] for i in range(action_probs_decode.shape[0])]
        else:
            predictions_list = [action_probs_decode]
        predicted_actions: List[List[str]] = []
        for predictions in predictions_list:
            argmax_indices = np.argmax(predictions, axis=-1)
            actions = []
            for a in argmax_indices:
                actions.append(str(Action(a).name))

            predicted_actions.append(actions)
        # print("predicted_actions:", predicted_actions)
        output_dict['predicted_actions'] = predicted_actions

        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = {}

        type_accuracy = self._type_accuracy.get_metric(reset)
        metric_dict['type_accuracy'] = type_accuracy
        #
        # for name, metric in self.type_f1_metrics.items():
        #     metric_val = metric.get_metric(reset)
        #     metric_dict[name + '_P'] = metric_val[0]
        #     metric_dict[name + '_R'] = metric_val[1]
        #     metric_dict[name + '_F1'] = metric_val[2]
        #
        # metric_dict['combined_metric'] = type_accuracy + \
        #                                  metric_dict['type_1_F1'] + metric_dict['type_2_F1'] + metric_dict['type_3_F1']
        return metric_dict



    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ProStructModel':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        use_attention = params.pop("use_attention")
        seq2seq_encoder_params = params.pop("seq2seq_encoder")
        seq2vec_encoder_params = params.pop("seq2vec_encoder")

        seq2seq_encoder = None
        seq2vec_encoder = None

        if use_attention:
            seq2seq_encoder = Seq2SeqEncoder.from_params(seq2seq_encoder_params)
        else:
            seq2vec_encoder = Seq2VecEncoder.from_params(seq2vec_encoder_params)

        # Initialize params for location span related layers
        span_end_encoder_after = Seq2SeqEncoder.from_params(params.pop("span_end_encoder_after"))

        initializer = InitializerApplicator.from_params(params.pop("initializer", []))
        use_decoder_trainer = params.pop("use_decoder_trainer")
        decoder_beam_search = None
        if use_decoder_trainer:
            decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))

        kb_configs = params.pop("kb_configs", {
            "kb_to_use": "lexicalkb",
            "lexical_kb_path": "tests/fixtures/decoder_data/kbs/kb3/lexical-kb-v0.tsv",
            "partial_grids_path": "tests/fixtures/decoder_data/kbs/kb2/kb2-partialgrids.tsv",
            "partialgrid_prompts_load_path": "tests/fixtures/decoder_data/kbs/kb2/partial-grids.tsv",
            "fullgrid_prompts_load_path": "tests/fixtures/decoder_data/kbs/kb2/full-grids.tsv"
        })

        # AllenNLP predictors requires no change in a serialized model
        # Making this more flexible by adding other_configs as a dict.
        other_configs = params.pop("other_configs", {})

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   seq2seq_encoder=seq2seq_encoder,
                   seq2vec_encoder=seq2vec_encoder,
                   use_attention=use_attention,
                   span_end_encoder_after=span_end_encoder_after,
                   use_decoder_trainer=use_decoder_trainer,
                   decoder_beam_search=decoder_beam_search,
                   kb_configs=kb_configs,
                   other_configs=other_configs,
                   initializer=initializer)
