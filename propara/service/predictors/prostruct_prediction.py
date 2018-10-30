from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor
from overrides import overrides
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from propara.data.propara_dataset_reader import Action
from propara.trainer_decoder.action_scorer import ActionScorerDummy


@Predictor.register('prostruct_prediction')
class ProStructPredictor(Predictor):
    """
    Wrapper for the :class:`processes.models.ProStructModel` model.
    This is used at prediction time, including on the demo when invoking the following command:
    # demo command:
    python  -m allennlp.service.server_simple
            --archive-path /tmp/xtiny/model.tar.gz
            --predictor prostruct_prediction
            --include-package processes
            --static-dir demo/propara_demo
    """

    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        # read one json instance from prostruct
        # sentence_texts: List[str]
        # participants: List[str],
        # states: List[List[str]], where states[i][j] is ith participant at time j

        # Para id is useful for decoder trainer. As we won't call it at prediction time,
        # we make this optional.
        para_id = inputs.get("para_id", -1)
        sentence_texts = inputs["sentence_texts"]
        sentence_texts = sentence_texts if "\n" not in sentence_texts else [s for s in sentence_texts.split("\n")]
        participants = inputs["participants"]
        if not participants:
            participants = [p for p in self.helper.participants_from_sentences(sentence_texts)]
        # Participants can be separated in many different ways
        # (A participant can contain comma and in those cases we separate by "\n" or "\t").
        # Do this only when participants is not already a list (demo passes a string).
        if isinstance(participants, str):
            if "\n" in participants:
                separator = "\n"
            elif "\t" in participants:
                separator = "\t"
            else:
                separator = ","
            participants = [p.strip() for p in participants.split(separator)]
            participants = participants if "," not in participants else [p.strip() for p in participants.split(",")]
        states = inputs.get("states", None)
        # Can be used in demo (eventually the demo would control more parameters such as which commonsense etc).
        top_k_sequences = inputs.get("top_k_sequences", 2)
        print(f"Predictor gets input: ", inputs)
        print(f"Predictor formats inputs =\n{para_id},\n{sentence_texts}\n{participants}")

        instance = self._dataset_reader.text_to_instance(para_id=para_id,
                                                         sentence_texts=sentence_texts,
                                                         participants=list(participants),
                                                         states=states,
                                                         filename="test"
                                                         # rules_activated="0,0,0,0"
                                                         )  # convert from set

        # Can we update instance based on self.proparaDecoderStep.update_rules()
        old_action_scorer = self._model.decoder_step.get_action_scorer()
        old_valid_action_gen = self._model.decoder_step.get_valid_action_generator()

        rules_used_original = self._model.decoder_step.get_valid_action_generator().get_rules_used()

        dont_use_kb = "dont_use_kb" in inputs and inputs["dont_use_kb"]
        if dont_use_kb:
            self._model.decoder_step.change_action_scorer(ActionScorerDummy())

        rules_changed = "rules_used" in inputs and inputs["rules_used"] is not None
        if rules_changed:
            updated_rules = [True if int(rule_val.strip()) > 0 else False
                             for rule_val in inputs["rules_used"].split(",")]
            self._model.decoder_step.get_valid_action_generator().set_rules_used(updated_rules)

        outputs = self._model.forward_on_instance(instance)

        # Reset to original settings.
        if dont_use_kb:
            self._model.decoder_step.change_action_scorer(old_action_scorer)
        if rules_changed:
            self._model.decoder_step.change_valid_action_generator(old_valid_action_gen)

        json_outputs = ProStructPredictor.to_json(
            outputs,
            participants,
            top_k_sequences
        )
        json_outputs["default_kb_used"] = self._model.decoder_step.get_action_scorer().name
        json_outputs["default_rules_used"] = rules_used_original
        json_outputs['predicted_locations'] = self.predict_locations(outputs, sentence_texts, participants)

        settings_used = ""
        if rules_changed or dont_use_kb:
            settings_used = f"rules used: {inputs.get('rules_used', '')} and using {'no kb' if dont_use_kb else 'kb'}"
        json_outputs['settings_used'] = settings_used

        json_outputs["sentences"] = sentence_texts
        return {**inputs, **json_outputs}

    def predict_locations(self,
                          outputs,
                          sentence_texts, #  sent
                          participants):  #  parti

        loc_per_sent_per_parti = []
        if 'location_span_after' not in outputs:
            return loc_per_sent_per_parti

        locs = outputs['location_span_after'][0] # Shape: (bs=1 always) sent x parti x span_start_end
        best_action_seq = outputs['best_final_states'][0][0].action_history[0] # step x participant_labels

        for sent_id, sent in enumerate(sentence_texts):
            loc_per_parti = []
            for parti_id, parti in enumerate(participants):
                start = locs[sent_id][parti_id][0].data[0]
                end = locs[sent_id][parti_id][1].data[0] # Inclusive
                # if action = create/move then output loc
                # if action = destroy then output '-'
                # if action = none then output '?'
                curr_action = best_action_seq[sent_id][parti_id]
                loc_per_parti.append(
                    self.span_from_sent(sent, start, end) if (curr_action == Action.CREATE.value or curr_action == Action.MOVE.value)
                    else (
                        '-' if curr_action == Action.DESTROY.value else '?'
                    )
                )
            loc_per_sent_per_parti.append(loc_per_parti)
        return loc_per_sent_per_parti

    def span_from_sent(self, sent, start, end):
        sent_tokens = self.tokenizer.tokenize(sent)
        return ' '.join([s.text for s in sent_tokens[start:(end+1)]]) if 0 <= start <= end < len(sent_tokens) else "?"

    @classmethod
    def to_json(cls, outputs, participants, top_k_sequences):
        """
            The predictor is expected to format outputs as json (simplifies rendering to UI server)
            Example of an "outputs" supplied to this function:
                For one instance with (4 participants 4 actions 10 steps),
                output from predictor looks like:
            {
            0: [   # 0 indicates first instance in the batch, we have a fixed batch size of 1.
                [
                    [(2, 1, 2, 0), (0, 1, 0, 1), ... (0, 0, 3, 0)]   # top 1st; 10 = number of steps.
                ] - 21.939434051513672, [
                    [(2, 0, 2, 1), (0, 1, 0, 1), ... (0, 0, 3, 0)]   # top 2nd
                ] - 21.940093994140625, [
                    [(2, 1, 2, 0), (0, 1, 0, 1), ... (0, 0, 3, 0)]   # top 3rd
                ...
                ]
                    [(2, 0, 2, 1), (0, 1, 0, 1), ... (0, 0, 0, 0)]   # top 10th
                ] - 21.94779396057129
            ]
           }
            """
        json = {}
        if outputs is not None:
            for rank, o in enumerate(outputs['best_final_states'][0][:top_k_sequences]):
                curr_top_rank = "top" + str(rank + 1)  # top1, top2...
                o_json = o.to_json()
                # o_json: {'action_history': [(0, 2, 0, 0, 2), ..., (0, 0, 0, 0, 0)], 'score': -33.67}
                # step: (0, 2, 0, 0, 2)
                # Replace index to Action (ignore zeros/NONE)
                json[curr_top_rank + "_labels"] = [[str(Action(int(label)).name).replace("NONE", "")
                                                    for label in step]
                                                   for step in o_json['action_history']
                                                   ]
                json[curr_top_rank + "_original"] = o_json['action_history']
                json[curr_top_rank + "_score"] = o_json['score']

        else:
            json["error_message"] = "No output predicted probably because " \
                                    "input participants is not found in paragraph."
        json["participants"] = participants
        return json
