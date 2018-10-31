class KB(object):

    def __init__(self):
        self.name = "kb source not filled in yet."

    # sample process_id: 5 (integer) from Propara
    # version 1.0 published with NAACL 2018 paper
    #
    # example of a cand: (snow, MOVE, ?, area, 0.691)
    # accepted variants: (snow, MOVE, , , )
    def prob_of(self, process_id, cand):
        raise NotImplementedError("Please Implement this method")


