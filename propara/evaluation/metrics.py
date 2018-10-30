class Metrics:

    def __init__(self):
        # true positive etc.
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0
        # # sometimes we need to average a score.
        # self.score_total = 0.0
        self.precision = -1.0
        self.recall = -1.0

    def tp_increment(self, increment_by):
        self.tp += increment_by

    def fp_increment(self, increment_by):
        self.fp += increment_by

    def tn_increment(self, increment_by):
        self.tn += increment_by

    def fn_increment(self, increment_by):
        self.fn += increment_by

    def set_precision(self, val):
        self.precision = val

    def set_recall(self, val):
        self.recall = val

    @staticmethod
    def compute_f1(prec, rec):
        if (prec + rec) != 0:
            return 2 * prec * rec / (prec + rec)
        else:
            return 0.0

    def get_scores(self):

        # Highway-- directly return results if there were already set.
        if self.precision >= 0.0:  # positive indicates it was set.
            # We don't know the accuracy so set it to -1.0
            return -1.0, self.precision, self.recall, Metrics.compute_f1(self.precision, self.recall)

        # Compute P, R, F1
        if self.tp + self.fp == 0:
            prec = 1.0
        else:
            prec = self.tp / (self.tp + self.fp)

        if self.tp + self.fn == 0:
            rec = 1.0
        else:
            rec = self.tp / (self.tp + self.fn)

        f1 = Metrics.compute_f1(prec, rec)

        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)
        return accuracy, prec, rec, f1

    def str(self):
        return str(self.get_scores()[3])
