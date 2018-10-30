from unittest import TestCase
import os

from propara.evaluation.eval import Evaluator


class TestEval(TestCase):
    def setUp(self):
        # Debugging from IDE requires ../fixtures/... path
        if os.path.exists('../fixtures/eval_data/eval_gold.tsv'):
            gold_file = '../fixtures/eval_data/eval_gold.tsv'
            system_file = '../fixtures/eval_data/eval_system.tsv'
        else:
            # pytest from CLI requires tests/fixtures/... path
            gold_file = 'tests/fixtures/eval_data/eval_gold.tsv'
            system_file = 'tests/fixtures/eval_data/eval_system.tsv'
        self.evaluator = Evaluator(system_file, gold_file)

    def test_load_results(self):
        # is the process id map correctly filled?
        assert self.evaluator.golds[101][2][0] == ['langur OR langer']  # tuple.
        assert self.evaluator.golds[101][1] == [['monkey'], ['ape']]  # tuple.
        assert self.evaluator.golds[100][4] == [['plant OR leaf', 'root', 'soil', 'event2']]
        assert self.evaluator.golds[100][3] == [['-']]  # tuple.
        assert self.evaluator.golds[101][3][0][0] == 'animal OR monkey'  # tuple.
        assert self.evaluator.systems[100][4][0][1] == 'root'  # tuple.
        assert self.evaluator.systems[101][3][0][0] == 'animal'  # tuple.
        assert self.evaluator.golds[101][3]  # is non null

    def test_match_score(self):
        assert self.evaluator.match_score('', '') == 1.0
        assert self.evaluator.match_score('', '-') == 0.0
        assert self.evaluator.match_score('?', '?') == 1.0
        assert self.evaluator.match_score('plant OR leaf', 'leaf') == 1.0
        assert self.evaluator.match_score('', 'leaf') == 0.0
        assert self.evaluator.match_score('-', 'leaf') == 0.0
        assert self.evaluator.match_score('plant  OR leaf', 'leaf') == 1.0
        assert self.evaluator.match_score('dew OR rain', 'water OR dew') == 1.0
        assert self.evaluator.match_score('dew', 'dew AND sun') == 0.5
        assert self.evaluator.match_score('dew AND sun', 'dew') == 0.5
        assert self.evaluator.match_score('dew AND sun', 'dew AND blah1 AND blah2') == 0.25
        assert self.evaluator.match_score('dew AND rain', 'water OR dew') == 0.5
        assert self.evaluator.match_score('water OR dew AND sun', 'dew OR rain') == 0.5

    # def test_score_nontuple_question(self):
    #     gold = [['water OR dew'], ['sun']]
    #     system = [['dew OR rain']]
    #     assert self.evaluator.score_question(gold, system) == 0.4

    def test_score_tuple_question(self):
        ques_id = 3
        golds = [['plant OR leaf', 'root', 'earth', 'event2'],
                 ['leaf', 'soil', 'mud', 'event2']]
        systems = [['plants OR leaf', 'root', 'earth', 'event2'],
                   ['plant', 'mud OR plant', 'room OR earth', 'event2']]
        systems_longer = [['plants OR leaf', 'root', 'earth', 'event2'],
                          ['plant', 'mud OR plant', 'room OR earth', 'event2'],
                          ['tree', 'monkey', 'earth', 'event2']]
        systems_shorter = [['plants OR leaf', 'root', 'earth', 'event2']]
        g_eq_len_s = self.evaluator.score_question(golds, systems)
        g_gt_len_s = self.evaluator.score_question(golds, systems_shorter)
        g_lt_len_s = self.evaluator.score_question(golds, systems_longer)
        g_gt_len_zero = self.evaluator.score_question(golds, [])
        g_zero_s = self.evaluator.score_question([], systems)
        g_zero_s_zero = self.evaluator.score_question([], [])
        assert g_eq_len_s.str() == "0.8333333333333333"
        assert g_gt_len_s.str() == "0.8"
        assert g_lt_len_s.str() == "0.6666666666666666"
        assert g_zero_s.str() == "0.0"
        assert g_gt_len_zero.str() == "0.0"
        assert g_zero_s_zero.str() == "1.0"

    def test_score_tuple(self):
        assert self.evaluator.score_tuple(
            self.evaluator.golds[101][3][0], self.evaluator.systems[101][3][0]) == 1.0
        assert self.evaluator.score_tuple(
            ['plant OR leaf', 'root', 'earth', 'event2'],
            ['leaf', 'root OR plant', 'soil OR earth', 'event2']) == 1.0
        # plants should match plant.
        assert self.evaluator.score_tuple(
            ['plants OR leaf', 'root', 'earth', 'event2'],
            ['plant', 'root OR plant', 'soil OR earth', 'event2']) == 1.0

    def test_score_answer_elem(self):
        assert self.evaluator.score_answer_elem([''], ['-']) == 0
        assert self.evaluator.score_answer_elem(['plant OR leaf'], ['leaf']) == 1
        assert self.evaluator.score_answer_elem(['?'], ['?']) == 1
        assert self.evaluator.score_answer_elem(
            self.evaluator.golds[101][3][0], self.evaluator.systems[101][3][0]) == 1.0
        assert self.evaluator.score_answer_elem(
            self.evaluator.golds[100][3][0], self.evaluator.systems[100][3][0]) == 0.0

    def test_score_question(self):
        metrics = self.evaluator.score_question(
            self.evaluator.golds[101][3], self.evaluator.systems[101][3])
        assert metrics.get_scores()[3] == 1.0
        metrics2 = self.evaluator.score_question(
            self.evaluator.golds[100][3], self.evaluator.systems[100][3])
        assert metrics2.get_scores()[3] == 0.0

    def test_score_all_questions(self):
        metrics_all = self.evaluator.score_all_questions()
        assert metrics_all[101][3].get_scores()[3] == 1.0
        assert metrics_all[100][3].get_scores()[3] == 0.0

    def test_score_empty_answers(self):
        metrics_all = self.evaluator.score_all_questions()
        for i in range(1, 4):
            assert metrics_all[102][i].get_scores()[3] == 1.0

    def test_gold_nonempty_system_empty(self):
        score = self.evaluator.score_tuple(['snow', 'mass', 'area', '4'], [''])
        assert (score == 0.0)
