from unittest import TestCase

from propara.evaluation.metrics import Metrics


class TestMetrics(TestCase):
    def setUp(self):
        self.metrics = Metrics()
        self.metrics.tp_increment(1)
        self.metrics.fp_increment(1)
        self.metrics.tp_increment(1)
        self.metrics.fn_increment(1)

        self.metrics_highway = Metrics()
        self.metrics_highway.set_precision(0.5)
        self.metrics_highway.set_recall(1.0)

    def test_get_scores(self):
        _2_by_3 = 0.6666666666666666
        assert self.metrics.get_scores() == (0.5, _2_by_3, _2_by_3, _2_by_3)
        assert self.metrics_highway.get_scores() == (-1.0, 0.5, 1.0, _2_by_3)

