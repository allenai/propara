from propara.trainer_decoder.propara_decoder_state import ProParaDecoderState

import torch


# The "tensor" flow in the DecoderTrainer model is:
#
# 1. model's logits of size `(batch_size * num_sentences * num_participants * num_actions)`
# are passed to decoder_trainer entry point.
#
# 2. reshaped to `(num_sentences * num_participants * num_actions)`
# because states are per instance.
#
# 3. these scores are modified (using background kb),
# shape still will be: `(num_sentences * num_participants * num_actions)`
#
# 4. but, the final score of a state requires to combine p1:score1, p2:score2
# in a state to score1+score2 -- this changes the shape to `(num_sentences)` .
# As we will do a squeeze(2).sum(-1), the flow of tensor won't break
# and hence is backprop-able.
#


class DecoderTrainerHelper:
    @classmethod
    # For modularity, plugging out the function from the forward() of ProParaBaselineModel
    # action_logits: bs * num_sentences * num_participants * num_actions
    # participant_list: bs, np, num_tokens_p, embedding_size
    def pass_on_info_to_decoder_trainer(
            cls,
            selfie,
            para_id_list,  # [batch_size] containing (int) para_ids.
            actions,  # This is gold sequence. (batch_size, num_sentences, num_participants, 1)
            target_mask,  # (batch_size, num_sentences, num_participants)
            participants_list,  # (batch_size, num_participants, embedding_dim)
            participant_strings,  # (batch_size, num_participants) List[List[str]]
            participant_indicators,  # (batch_size *  num_parti * num_sent * sent_len) List[List[List[List[int]]]]
            logit_tensor  # (batch_size, num_sentences, num_participants, num_actions)
    ):
        # Depending on the type of DecoderTrainer we use, there will be different types of supervision.
        # ERM's supervision = cost function, not any explicit examples from training set.
        # MML uses training examples, where one or more sequences are gold.
        # In our case, gold has exactly one sequence.
        if actions is not None:
            # actions sequence should be (batch_size, num_valid_sequences, num_sentences, num_participants)
            # here we have one valid sequence, so we need to reshape things:
            target_action_sequences = actions.squeeze(-1).unsqueeze(1)
            # target_mask needs to have the same shape, but is currently (batch_size, num_sentences, num_participants)
            target_mask = target_action_sequences != -1
        else:
            target_mask = None

        batch_size = participants_list.shape[0]
        num_participants = participants_list.shape[1]
        selfie.decoder_step.set_num_participants(num_participants)
        selfie.decoder_step.set_num_steps(logit_tensor.shape[1])

        # The decoder trainer can work with one instance at a time.
        # Iterate over the instances in the batch and aggregate total loss.
        # print("in helper: training set: calling decoder_trainer")
        total_loss = 0.0
        best_final_states = []

        for instance_id in range(0, batch_size):
            initial_state = ProParaDecoderState(
                # grouped_state with group size 1
                group_indices=[0],
                # history for group element 0
                action_history=[[]],
                # shape (num_groups, num_participants)
                participant_score_per_group=[
                    torch.autograd.Variable(
                        logit_tensor.data.new(num_participants).fill_(0.0)
                    )
                ],
                # commonsense based scorer requires embedding of the participants to
                # compute problem of an action for a participant
                # e.g., (power plants do not move).
                participants_embedding=[participants_list[instance_id]],
                # from: bs * num_sentences * num_participants * num_actions
                # to: num_sentences * num_participants * num_actions
                # This is used internally to compute the total score per
                # participant. action_logits do not carry forward in loss.
                # action_logits=actions_list[instance_id],
                # To backprop, the flow of tensor from prostruct model cannot
                # be broken (e.g., by pulling out scores like in action_logits)
                # We will modify this logit_tensor all the way through.
                logit_tensor=logit_tensor,
                # We need this to update scores for a particular instance in the
                # batch that the logit_tensor represents.
                instance_id=instance_id,
                # commonsense based scorer requires the process id/ topic and the
                # participant values are temporarily used.
                metadata={'process_id': para_id_list[instance_id],
                          'participant_strings': participant_strings[instance_id],
                          'participant_indicators': participant_indicators[instance_id]
                          },
                overall_score=None
            )

            not_in_test = (selfie.training or 'test' not in selfie.filename) and actions is not None

            if not_in_test:
                initial_state.metadata['in_beam_search'] = False
                # Aggregate the loss over all the instances in the batch.
                cur_output_dict = selfie.decoder_trainer.decode(
                    initial_state,
                    selfie.decoder_step,
                    # Supervision can also include rewards.
                    (target_action_sequences, target_mask),
                    selfie.instance_score or None
                )
                total_loss += cur_output_dict['loss']

            initial_state.metadata['in_beam_search'] = True
            best_final_states.append(selfie.beam_search.search(selfie.num_sentences,
                                                          initial_state,
                                                          selfie.decoder_step,
                                                          keep_final_unfinished_states=False))

        result = {'best_final_states': best_final_states}

        # At test time do not pass loss.
        if not_in_test:
            result['loss'] = total_loss

        return result
