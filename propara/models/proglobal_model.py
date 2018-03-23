from typing import Dict, List

import torch
from torch.autograd import Variable
from torch.nn.functional import nll_loss
from torch.nn.functional import softmax

import numpy as np

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator

@Model.register("ProGlobal")
class ProGlobal(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pos_field_embedder: TextFieldEmbedder,
                 sent_pos_field_embedder: TextFieldEmbedder,
                 modeling_layer: Seq2SeqEncoder,
                 span_end_encoder_before: Seq2SeqEncoder,
                 span_start_encoder_after: Seq2SeqEncoder,
                 span_end_encoder_after: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:

        """
        This ``Model`` takes as input a dataset read by ProGlobalDatasetReader
        Input: a list of sentences, a participant
        Output: location category for the participant, location span
        The basic outline of this model is to
            1. get an embedded representation for paragraph tokens,
            2. apply bi-LSTM to get contextual embeddings,
            3. apply three-category classification and location span prediction to predict the location state

        :param vocab: Vocabulary
        :param text_field_embedder: ``TextFieldEmbedder`` used to embed the ``sentence tokens``
        :param pos_field_embedder: ``TextFieldEmbedder``  used to embed the ``word positions``
        :param sent_pos_field_embedder: ``TextFieldEmbedder`` used to embed the sent indicators
        :param modeling_layer:  ``Seq2SeqEncoder``  to encode the sequence of paragraph
        :param span_end_encoder_bef: ``Seq2SeqEncoder`` to encode the sequence for before location end prediction
        :param span_start_encoder_aft: ``Seq2SeqEncoder`` to encode the sequence for after location start prediction
        :param span_end_encoder_aft: ``Seq2SeqEncoder`` to encode the sequence for after location end prediction
        :param dropout:
        :param mask_lstms:
        :param initializer: ``InitializerApplicator`` We will use this to initialize the parameters in the model

        Sample commandline
        ------------------
        python processes/run.py train -s /output_folder experiment_config/ProGlobal.json
        """

        super(ProGlobal, self).__init__(vocab)

        # embedders for text, word positions, and sentence indicators
        self._text_field_embedder = text_field_embedder
        self._pos_field_embedder = pos_field_embedder
        self._sent_pos_field_embedder = sent_pos_field_embedder

        # bi-LSTM: to generate the contextual embedding
        self._modeling_layer = modeling_layer
        modeling_dim = modeling_layer.get_output_dim()

        # three category classifier for before location
        self._category_before_predictor = torch.nn.Linear(modeling_dim, 3)

        # LSTM encoder for before location end: encode the contextual embedding and before location start scores
        self._span_end_encoder_before = span_end_encoder_before

        # linear function for before location start
        span_start_before_input_dim = modeling_dim
        self._span_start_predictor_before = TimeDistributed(torch.nn.Linear(span_start_before_input_dim, 1))

        # linear function for before location end
        span_end_before_encoding_dim = span_end_encoder_before.get_output_dim()
        span_end_before_input_dim = modeling_dim + span_end_before_encoding_dim
        self._span_end_predictor_before = TimeDistributed(torch.nn.Linear(span_end_before_input_dim, 1))

        # three category classifier for after location
        self._category_after_predictor = torch.nn.Linear(modeling_dim+3, 3)

        # LSTM encoder for after location start: encode the contextual embedding and
        # previous before location start scores
        self._span_start_encoder_after = span_start_encoder_after

        # linear function for after location start
        span_start_after_encoding_dim = span_start_encoder_after.get_output_dim()
        span_start_after_input_dim = modeling_dim + span_start_after_encoding_dim
        self._span_start_predictor_after = TimeDistributed(torch.nn.Linear(span_start_after_input_dim, 1))

        # LSTM encoder for after location end: encode the contextual embedding and
        # current before location start scores
        self._span_end_encoder_after = span_end_encoder_after
        span_end_after_encoding_dim = span_end_encoder_after.get_output_dim()
        span_end_after_input_dim = modeling_dim + span_end_after_encoding_dim

        # linear function for after location end
        self._span_end_predictor_after = TimeDistributed(torch.nn.Linear(span_end_after_input_dim, 1))

        self._dropout = torch.nn.Dropout(p=dropout)
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, tokens_list: Dict[str, torch.LongTensor], positions_list: Dict[str, torch.LongTensor],
                sent_positions_list: Dict[str, torch.LongTensor],
                before_loc_start: torch.IntTensor = None, before_loc_end: torch.IntTensor = None,
                after_loc_start_list: torch.IntTensor = None, after_loc_end_list: torch.IntTensor = None,
                before_category: torch.IntTensor = None, after_category_list: torch.IntTensor = None,
                before_category_mask: torch.IntTensor = None, after_category_mask_list: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:

        """
        :param tokens_list: Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        :param positions_list: same as tokens_list
        :param sent_positions_list: same as tokens_list
        :param before_loc_start: torch.IntTensor = None, required
            An integer ``IndexField`` representation of the before location start
        :param before_loc_end: torch.IntTensor = None, required
            An integer ``IndexField`` representation of the before location end
        :param after_loc_start_list: torch.IntTensor = None, required
            A list of integers ``ListField (IndexField)`` representation of the list of after location starts
            along the sequence of steps
        :param after_loc_end_list: torch.IntTensor = None, required
            A list of integers ``ListField (IndexField)`` representation of the list of after location ends
            along the sequence of steps
        :param before_category: torch.IntTensor = None, required
            An integer ``IndexField`` representation of the before location category
        :param after_category_list: torch.IntTensor = None, required
            A list of integers ``ListField (IndexField)`` representation of the list of after location categories
            along the sequence of steps
        :param before_category_mask: torch.IntTensor = None, required
            An integer ``IndexField`` representation of whether the before location is known or not (0/1)
        :param after_category_mask_list: torch.IntTensor = None, required
            A list of integers ``ListField (IndexField)`` representation of the list of whether after location is
            known or not for each step along the sequence of steps
        :return:
        An output dictionary consisting of:
        best_span: torch.FloatTensor
            A tensor of shape ``()``
        true_span: torch.FloatTensor
        loss: torch.FloatTensor
        """

        # batchsize * listLength * paragraphSize * embeddingSize
        input_embedding_paragraph = self._text_field_embedder(tokens_list)
        input_pos_embedding_paragraph = self._pos_field_embedder(positions_list)
        input_sent_pos_embedding_paragraph = self._sent_pos_field_embedder(sent_positions_list)
        # batchsize * listLength * paragraphSize * (embeddingSize*2)
        embedding_paragraph = torch.cat([input_embedding_paragraph, input_pos_embedding_paragraph,
                                input_sent_pos_embedding_paragraph], dim=-1)

        # batchsize * listLength * paragraphSize,  this mask is shared with the text fields and sequence label fields
        para_mask = util.get_text_field_mask(tokens_list, num_wrapping_dims=1).float()

        # batchsize * listLength ,  this mask is shared with the index fields
        para_index_mask, para_index_mask_indices = torch.max(para_mask, 2)

        # apply mask to update the index values,  padded instances will be 0
        after_loc_start_list = (after_loc_start_list.float() * para_index_mask.unsqueeze(2)).long()
        after_loc_end_list = (after_loc_end_list.float() * para_index_mask.unsqueeze(2)).long()
        after_category_list = (after_category_list.float() * para_index_mask.unsqueeze(2)).long()
        after_category_mask_list = (after_category_mask_list.float() * para_index_mask.unsqueeze(2)).long()

        batch_size, list_size, paragraph_size, input_dim = embedding_paragraph.size()

        # to store the values passed to next step
        tmp_category_probability = torch.zeros(batch_size, 3)
        tmp_start_probability = torch.zeros(batch_size, paragraph_size)

        loss = 0

        # store the predict logits for the whole lists
        category_predict_logits_after_list = torch.rand(batch_size, list_size, 3)
        best_span_after_list = torch.rand(batch_size, list_size, 2)

        for index in range(list_size):
            # get one slice of step for prediction
            embedding_paragraph_slice = embedding_paragraph[:, index, :, :].squeeze(1)
            para_mask_slice = para_mask[:, index, :].squeeze(1)
            para_lstm_mask_slice = para_mask_slice if self._mask_lstms else None
            para_index_mask_slice = para_index_mask[:, index]
            after_category_mask_slice = after_category_mask_list[:, index, :].squeeze()

            # bi-LSTM: generate the contextual embeddings for the current step
            # size: batchsize * paragraph_size * modeling_layer_hidden_size
            encoded_paragraph = self._dropout(self._modeling_layer(embedding_paragraph_slice, para_lstm_mask_slice))

            # max-pooling output for three category classification
            category_input, category_input_indices = torch.max(encoded_paragraph, 1)

            modeling_dim = encoded_paragraph.size(-1)
            span_start_input = encoded_paragraph

            # predict the initial before location state
            if index == 0:

                # three category classification for initial before location
                category_predict_logits_before = self._category_before_predictor(category_input)
                tmp_category_probability = category_predict_logits_before

                '''Model the before_loc prediction'''
                # predict the initial before location start scores
                # shape:  batchsize * paragraph_size
                span_start_logits_before = self._span_start_predictor_before(span_start_input).squeeze(-1)
                # shape:  batchsize * paragraph_size
                span_start_probs_before = util.masked_softmax(span_start_logits_before, para_mask_slice)
                tmp_start_probability = span_start_probs_before

                # shape:  batchsize * hiddensize
                span_start_representation_before = util.weighted_sum(encoded_paragraph, span_start_probs_before)

                # Shape: (batch_size, passage_length, modeling_dim)
                tiled_start_representation_before = span_start_representation_before.unsqueeze(1).expand(batch_size,
                                                                                                   paragraph_size,
                                                                                                   modeling_dim)

                # incorporate the original contextual embeddings and weighted sum vector from location start prediction
                # shape: batchsize * paragraph_size * 2hiddensize
                span_end_representation_before = torch.cat([encoded_paragraph,
                                                            tiled_start_representation_before], dim=-1)
                # Shape: (batch_size, passage_length, encoding_dim)
                encoded_span_end_before = self._dropout(
                    self._span_end_encoder_before(span_end_representation_before, para_lstm_mask_slice))

                # initial before location end prediction
                encoded_span_end_before = torch.cat([encoded_paragraph, encoded_span_end_before], dim=-1)
                # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
                span_end_logits_before = self._span_end_predictor_before(encoded_span_end_before).squeeze(-1)
                span_end_probs_before = util.masked_softmax(span_end_logits_before, para_mask_slice)

                # best_span_bef = self._get_best_span(span_start_logits_bef, span_end_logits_bef)
                best_span_before, best_span_before_start, best_span_before_end, best_span_before_real = \
                    self._get_best_span_single_extend(span_start_logits_before, span_end_logits_before,
                                                      category_predict_logits_before, before_category_mask)

                # compute the loss for initial bef location three-category classification
                before_null_pred = softmax(category_predict_logits_before)
                before_null_pred_values, before_null_pred_indices = torch.max(before_null_pred, 1)
                loss += nll_loss(before_null_pred, before_category.squeeze(-1))

                # compute the loss for initial bef location start/end prediction
                before_loc_start_pred = util.masked_softmax(span_start_logits_before, para_mask_slice)
                logpy_before_start = torch.gather(before_loc_start_pred, 1, before_loc_start).view(-1).float()
                before_category_mask = before_category_mask.float()
                loss += -(logpy_before_start * before_category_mask).mean()
                before_loc_end_pred = util.masked_softmax(span_end_logits_before, para_mask_slice)
                logpy_before_end = torch.gather(before_loc_end_pred, 1, before_loc_end).view(-1)
                loss += -(logpy_before_end * before_category_mask).mean()

                # get the real predicted location spans
                # convert category output (Null and Unk) into spans ((-2,-2) or (-1, -1))
                before_loc_start_real = self._get_real_spans_extend(before_loc_start, before_category,
                                                                    before_category_mask)
                before_loc_end_real = self._get_real_spans_extend(before_loc_end, before_category,
                                                                  before_category_mask)
                true_span_before = torch.stack([before_loc_start_real, before_loc_end_real], dim=-1)
                true_span_before = true_span_before.squeeze(1)

            # input for (after location) three category classification
            category_input_after = torch.cat((category_input, tmp_category_probability), dim=1)
            category_predict_logits_after = self._category_after_predictor(category_input_after)
            tmp_category_probability = category_predict_logits_after

            # copy the predict logits for the index of the list
            category_predict_logits_after_tmp = category_predict_logits_after.unsqueeze(1)
            category_predict_logits_after_list[:, index, :] = category_predict_logits_after_tmp.data

            '''  Model the after_loc prediction  '''
            # after location start prediction: takes contextual embeddings and weighted sum vector as input
            # shape:  batchsize * hiddensize
            prev_start = util.weighted_sum(category_input, tmp_start_probability)
            tiled_prev_start = prev_start.unsqueeze(1).expand(batch_size, paragraph_size, modeling_dim)
            span_start_input_after = torch.cat((span_start_input, tiled_prev_start), dim=2)
            encoded_start_input_after = self._dropout(
                self._span_start_encoder_after(span_start_input_after, para_lstm_mask_slice))
            span_start_input_after_cat = torch.cat([encoded_paragraph, encoded_start_input_after], dim=-1)

            # predict the after location start
            span_start_logits_after = self._span_start_predictor_after(span_start_input_after_cat).squeeze(-1)
            # shape:  batchsize * paragraph_size
            span_start_probs_after = util.masked_softmax(span_start_logits_after, para_mask_slice)
            tmp_start_probability = span_start_probs_after

            # after location end prediction: takes contextual embeddings and weight sum vector as input
            # shape:  batchsize * hiddensize
            span_start_representation_after = util.weighted_sum(encoded_paragraph, span_start_probs_after)
            # Tensor Shape: (batch_size, passage_length, modeling_dim)
            tiled_start_representation_after = span_start_representation_after.unsqueeze(1).expand(batch_size,
                                                                                           paragraph_size,
                                                                                           modeling_dim)
            # shape: batchsize * paragraph_size * 2hiddensize
            span_end_representation_after = torch.cat([encoded_paragraph, tiled_start_representation_after], dim=-1)
            # Tensor Shape: (batch_size, passage_length, encoding_dim)
            encoded_span_end_after = self._dropout(self._span_end_encoder_after(span_end_representation_after,
                                                                                para_lstm_mask_slice))
            encoded_span_end_after = torch.cat([encoded_paragraph, encoded_span_end_after], dim=-1)
            # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
            span_end_logits_after = self._span_end_predictor_after(encoded_span_end_after).squeeze(-1)
            span_end_probs_after = util.masked_softmax(span_end_logits_after, para_mask_slice)

            # get the best span for after location prediction
            best_span_after, best_span_after_start, best_span_after_end, best_span_after_real = \
                self._get_best_span_single_extend(span_start_logits_after, span_end_logits_after,
                                                  category_predict_logits_after, after_category_mask_slice)

            # copy current best span to the list for final evaluation
            best_span_after_list[:, index, :] = best_span_after.data.view(batch_size, 1, 2)

            """ Compute the Loss for this slice """
            after_category_mask = after_category_mask_slice.float().squeeze(-1)  # batchsize
            after_category_slice = after_category_list[:, index, :]  # batchsize * 1
            after_loc_start_slice = after_loc_start_list[:, index, :]
            after_loc_end_slice = after_loc_end_list[:, index, :]

            # compute the loss for (after location) three category classification
            para_index_mask_slice_tiled = para_index_mask_slice.unsqueeze(1).expand(para_index_mask_slice.size(0), 3)
            after_category_pred = util.masked_softmax(category_predict_logits_after, para_index_mask_slice_tiled)
            logpy_after_category = torch.gather(after_category_pred, 1, after_category_slice).view(-1)
            loss += -(logpy_after_category * para_index_mask_slice).mean()

            # compute the loss for location start/end prediction
            after_loc_start_pred = util.masked_softmax(span_start_logits_after, para_mask_slice)
            logpy_after_start = torch.gather(after_loc_start_pred, 1, after_loc_start_slice).view(-1)
            loss += -(logpy_after_start * after_category_mask).mean()
            after_loc_end_pred = util.masked_softmax(span_end_logits_after, para_mask_slice)
            logpy_after_end = torch.gather(after_loc_end_pred, 1, after_loc_end_slice).view(-1)
            loss += -(logpy_after_end * after_category_mask).mean()

        # for evaluation  (combine the all annotations)
        after_loc_start_real = self._get_real_spans_extend_list(after_loc_start_list, after_category_list,
                                                                after_category_mask_list)
        after_loc_end_real = self._get_real_spans_extend_list(after_loc_end_list, after_category_list,
                                                              after_category_mask_list)

        true_span_after = torch.stack([after_loc_start_real, after_loc_end_real], dim=-1)
        true_span_after = true_span_after.squeeze(2)
        best_span_after_list = Variable(best_span_after_list)

        true_span_after = true_span_after.view(true_span_after.size(0) * true_span_after.size(1),
                                               true_span_after.size(2)).float()

        para_index_mask_tiled = para_index_mask.view(-1, 1)
        para_index_mask_tiled = para_index_mask_tiled.expand(para_index_mask_tiled.size(0), 2)

        para_index_mask_tiled2 = para_index_mask.unsqueeze(2).expand(para_index_mask.size(0),
                                                                     para_index_mask.size(1), 2)
        after_category_mask_list_tiled = after_category_mask_list.expand(batch_size, list_size, 2)
        after_category_mask_list_tiled = after_category_mask_list_tiled*para_index_mask_tiled2.long()

        # merge all the best spans predicted for the current batch, filter out the padded instances
        merged_sys_span, merged_gold_span = self._get_merged_spans(true_span_before, best_span_before, true_span_after,
                                                                   best_span_after_list, para_index_mask_tiled)

        output_dict = {}
        output_dict["best_span"] = merged_sys_span.view(1, merged_sys_span.size(0)*merged_sys_span.size(1))
        output_dict["true_span"] = merged_gold_span.view(1, merged_gold_span.size(0)*merged_gold_span.size(1))
        output_dict["loss"] = loss
        return output_dict

    # merge system spans and gold spans for a batchsize of lists, based on mask
    def _get_merged_spans(self, gold_span_before: Variable, sys_span_before: Variable, gold_span_after: Variable,
                          sys_span_after: Variable, mask: Variable):
        batchsize, listsize, d = sys_span_after.size()
        gold_span_before = gold_span_before.numpy()
        gold_span_after = gold_span_after.numpy()
        sys_span_before = sys_span_before.data.cpu().numpy()
        sys_span_after = sys_span_after.data.cpu().numpy()
        mask = mask.data.cpu().numpy()
        merged_sys_span = []
        merged_gold_span = []
        for i in range(batchsize):
            merged_sys_span.append(sys_span_before[i])
            merged_gold_span.append(gold_span_before[i])
            for j in range(listsize):
                if mask[i*listsize+j][0]==1:
                    merged_sys_span.append(sys_span_after[i][j])
                    merged_gold_span.append(gold_span_after[i*listsize+j])

        merged_sys_span_new = np.zeros((len(merged_sys_span), 2), dtype=np.long)
        merged_gold_span_new = np.zeros((len(merged_gold_span), 2), dtype=np.long)

        for i in range(len(merged_sys_span)):
            tmp = merged_sys_span[i]
            merged_sys_span_new[i] = tmp
            tmp1 = merged_gold_span[i]
            merged_gold_span_new[i] = tmp1
        merged_sys_span = torch.from_numpy(merged_sys_span_new)
        merged_gold_span = torch.from_numpy(merged_gold_span_new)
        return merged_sys_span, merged_gold_span

    # convert null to -2, unk to -1, return all the location spans
    def _get_real_spans_extend(self, loc_anno: Variable, category_anno: Variable, category_mask: Variable):
        batch_size, v = loc_anno.size()

        real_loc_anno = np.zeros((batch_size, v), dtype=np.long)
        loc_anno = loc_anno.data.cpu().numpy()
        category_anno = category_anno.data.cpu().numpy()

        for b in range(batch_size):
            if category_anno[b, 0] == 1:
                real_loc_anno[b, 0] = -1
            elif category_anno[b, 0] == 2:
                real_loc_anno[b, 0] = -2
            elif category_anno[b, 0] == 0:
                real_loc_anno[b, 0] = loc_anno[b, 0]
        real_loc_anno = torch.from_numpy(real_loc_anno)
        return real_loc_anno

    # convert null to -2, unk to -1, return all the location spans
    def _get_real_spans_extend_list(self, loc_anno: Variable, category_anno: Variable, category_mask: Variable):
        batch_size, list_size, v = loc_anno.size()

        real_loc_anno = np.zeros((batch_size, list_size, v), dtype=np.long)
        loc_anno = loc_anno.data.cpu().numpy()       # batch_size * list_size * 1
        category_anno = category_anno.data.cpu().numpy()

        for b in range(batch_size):
            for l in range(list_size):
                if category_anno[b, l, 0] == 1:
                    real_loc_anno[b, l, 0] = -1
                elif category_anno[b, l, 0] == 2:
                    real_loc_anno[b, l, 0] = -2
                elif category_anno[b, l, 0] == 0:
                    real_loc_anno[b, l, 0] = loc_anno[b, l, 0]
        real_loc_anno = torch.from_numpy(real_loc_anno)
        return real_loc_anno

    # convert null to -2, unk to -1, return all the location spans
    def _get_best_span_single_extend(self, span_start_logits: Variable, span_end_logits: Variable,
                                     category_predict_logits: Variable, category_mask: Variable):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()

        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long()

        best_start_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size).fill_(0)).float()

        best_end_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size).fill_(0)).float()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()
        category_predict_logits = category_predict_logits.data.cpu().numpy()
        category_mask = category_mask.data.cpu().numpy()
        category_best_pos = np.argmax(category_predict_logits, axis=1)

        real_loc_size = 0
        for i in range(batch_size):
            if category_mask[i]==1:
                real_loc_size = real_loc_size+1
        real_best_word_span = Variable(torch.rand(real_loc_size, 2).fill_(0)).long()

        real_index = 0
        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                span_length = j - span_start_argmax[b]
                if val1 + val2 > max_span_log_prob[b] and span_length < 6:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2

            if category_best_pos[b] == 1:
                best_word_span[b, 0] = -1
                best_word_span[b, 1] = -1
            elif category_best_pos[b] == 2:
                best_word_span[b, 0] = -2
                best_word_span[b, 1] = -2
            if category_mask[b] == 1:
                real_best_word_span[real_index, 0] = best_word_span[b, 0]
                real_best_word_span[real_index, 1] = best_word_span[b, 1]
                real_index = real_index+1
            best_start_span[b] = best_word_span[b, 0]
            best_end_span[b] = best_word_span[b, 1]
        return best_word_span, best_start_span, best_end_span, real_best_word_span

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                }

    # to get the best span based on location start and location end scores (maximal answer length is 5)
    def _get_best_span(self, span_start_logits: Variable, span_end_logits: Variable) -> Variable:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                # get the current max value till j
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                # end value should start from j
                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ProGlobal':
        token_embedder_params = params.pop("text_field_embedder")
        pos_embedder_params = params.pop("pos_field_embedder")
        sent_pos_embedder_params = params.pop("sent_pos_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, token_embedder_params)
        pos_field_embedder = TextFieldEmbedder.from_params(vocab, pos_embedder_params)
        sent_pos_field_embedder = TextFieldEmbedder.from_params(vocab, sent_pos_embedder_params)

        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        span_end_encoder_before = Seq2SeqEncoder.from_params(params.pop("span_end_encoder_bef"))
        span_start_encoder_after = Seq2SeqEncoder.from_params(params.pop("span_start_encoder_aft"))
        span_end_encoder_after = Seq2SeqEncoder.from_params(params.pop("span_end_encoder_aft"))
        dropout = params.pop('dropout', 0.2)

        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   pos_field_embedder=pos_field_embedder,
                   sent_pos_field_embedder=sent_pos_field_embedder,
                   modeling_layer=modeling_layer,
                   span_start_encoder_after=span_start_encoder_after,
                   span_end_encoder_before=span_end_encoder_before,
                   span_end_encoder_after=span_end_encoder_after,
                   dropout=dropout,
                   initializer=initializer)
