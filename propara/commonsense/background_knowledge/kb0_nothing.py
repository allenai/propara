from propara.commonsense import model_output_loader
from propara.commonsense.background_knowledge.kb import KB


class KB0Nothing(KB):

    def __init__(self):
        super().__init__()
        self.name = 'kb_none'

    def prob_of(self, process_id, cand):
        return 1.0
