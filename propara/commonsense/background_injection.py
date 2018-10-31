#  1. model output: predictions per step in diff. paragraphs.
#  2. input commonsense based on para_id.
#  3. (resulting input to beam) score per step
#      based on model score and paragraph coherence.
from propara.commonsense import model_output_loader
from propara.commonsense.background_knowledge.kb import KB


class BackgroundInjection:
    def __init__(self, background_kb: KB):
        self.background_kb = background_kb

    def score_background(self, process_id, step_id, cand_id):
        cand = model_output_loader.step_cand(process_id, step_id, cand_id)
        proba = self.background_kb.prob_of(process_id, cand)
        # proba = kb2_partialgrids.prob_of(process_id, cand)
        ## proba = kb1_goldgrids.prob_of(process_id, cand)
        return proba

    def score_model(self, process_id, step_id, cand_id):
        cand = model_output_loader.step_cand(process_id, step_id, cand_id)
        score = cand[model_output_loader.IDX_SCORE] if len(cand) == model_output_loader.EXPECTED_LEN_CAND else 0.0
        return score

    def score_overall(self, process_id, step_id, cand_id):
        s = self.score_background(process_id, step_id, cand_id) * \
            self.score_model(process_id, step_id, cand_id)
        # returning zeros causes -log(0) problems in beam search.
        return 1e-20 if s == 0 else s

    # num_classes >= beam_size
    def matrix_for_beam(self, process_id, num_classes):
        # sample_beam_data = [
        #         [0.1, 0.2, 0.3, 0.4, 0.5],
        #         [0.5, 0.4, 0.3, 0.2, 0.1]
        # ]

        beam_data = []
        for step_id in model_output_loader.step_ids(process_id):
            step_data = []
            for cand_id in range(0, num_classes):
                step_score = self.score_overall(process_id, step_id, cand_id)
                step_data.append(step_score)
            beam_data.append(step_data)
        return beam_data
