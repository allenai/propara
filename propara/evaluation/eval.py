import logging

from processes.evaluation.metrics import Metrics
from nltk import PorterStemmer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluator:
    # system_file = '../fixtures/eval_data/eval_system.tsv'
    def __init__(self, system_file, gold_file='../../tests/fixtures/eval_data/eval_gold.tsv'):
        self.systems: dict = self.load_results(system_file)
        self.golds: dict = self.load_results(gold_file)
        # print("Evaluator: Debug: Loaded golds with ", len(self.golds.keys()),
        #       "keys\nsystem with ", len(self.systems.keys()))

    # Loads results from a file into a dictionary
    # process_id ->
    #               ques_id -> [answers]
    # where one answer is a tuple e.g.:
    # [init_value, final_value, loc_value, step_value]
    #
    # Specific input file format =
    # processid TAB    quesid TAB   answer_tsv
    #   multiple answers separated by `tab`,
    #   slots within each answer separated by `++++`
    def load_results(self, from_file_path: str) -> dict:
        logger.info("Evaluation: Loading answers from: %s", from_file_path)
        results = dict()
        with open(from_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                # process_id TAB    ques_id TAB   answer_tsv
                cols = line.split('\t')
                process_id = int(cols[0])
                if process_id not in results.keys():
                    results.setdefault(process_id, dict())
                ques_id = int(cols[1])
                answers = cols[2:]
                # empty answers should not be washed away.
                if len(answers) == 0:
                    answers = [""]
                # insert answers
                if ques_id not in results[process_id].keys():
                    results[process_id].setdefault(ques_id, [])
                for a in answers:
                    # slots within an answer sep. by `++++`
                    # form a tuple.
                    answer_slots = a.split('++++')
                    results[process_id][ques_id].append(answer_slots)
        return results

    @classmethod
    def stem(cls, w: str):
        if not w or len(w.strip()) == 0:
            return ""
        w_lower = w.lower()
        # Remove leading articles from the phrase (e.g., the rays => rays).
        # FIXME: change this logic to accept a list of leading articles.
        if w_lower.startswith("a "):
            w_lower = w_lower[2:]
        elif w_lower.startswith("an "):
            w_lower = w_lower[3:]
        elif w_lower.startswith("the "):
            w_lower = w_lower[4:]
        elif w_lower.startswith("your "):
            w_lower = w_lower[5:]
        elif w_lower.startswith("his "):
            w_lower = w_lower[4:]
        elif w_lower.startswith("their "):
            w_lower = w_lower[6:]
        elif w_lower.startswith("my "):
            w_lower = w_lower[3:]
        elif w_lower.startswith("another "):
            w_lower = w_lower[8:]
        elif w_lower.startswith("other "):
            w_lower = w_lower[6:]
        elif w_lower.startswith("this "):
            w_lower = w_lower[5:]
        elif w_lower.startswith("that "):
            w_lower = w_lower[5:]
        # Porter stemmer: rays => ray
        return PorterStemmer().stem(w_lower).strip()

    @classmethod
    def stem_words(cls, words):
        return [Evaluator.stem(w) for w in words]

    @classmethod
    def extract_answers(cls, answer_str):
        answers_outer = [g.strip() for g in answer_str.split(" AND ")]
        # Remaining answers can contain OR. So further split them to create a List of [ORed_answers]
        # answers_inner example: [dew,rain][sun] (comma indicates OR)
        answers_inner = [set(Evaluator.stem_words(outer.split(" OR "))) for outer in answers_outer]
        # Stem and remove leading stop words.
        return answers_inner

    def match_score(self, gold_answer_str: str, system_answer_str: str) -> float:
        # examples of answer_str: "water OR dew", "water", "water AND sun", "water OR H20 AND sun"
        # Note: gold and system answers can contain AND (partial scores here.),
        #       gold and system answers can contain OR (no partial scoring here.)
        #       both AND, OR cannot be present in one answer_str.

        # Trivial match (they can both be blank, ? or -).
        if gold_answer_str == system_answer_str:
            return 1.0

        system_answers = Evaluator.extract_answers(system_answer_str)
        gold_answers = Evaluator.extract_answers(gold_answer_str)

        # Compute overlap score.
        # system_answers = [sa1, sa2, ...] (these are AND values essentially)
        # sa1 example: set(water, dew)
        # if sa1 is found in any gold, then sa1_score = 1.0 else 0.0
        # Final score = avg_i (sa_i).
        s_intersection_gold_set = [0.0] * len(system_answers)
        for sa_idx, sa in enumerate(system_answers):
            # qa example: set(water, rain)
            sa_found_match = False
            for ga in gold_answers:
                if len(sa.intersection(ga)) > 0:
                    sa_found_match = True
                    break
            s_intersection_gold_set[sa_idx] = 1.0 if sa_found_match else 0.0

        # Final score is jaccard like = (A intersection B) / (A union B)
        # A intersection B
        numerator = sum(s_intersection_gold_set)
        # A union B = A + B - A intersection B
        denominator = 1.0 * (len(system_answers) + len(gold_answers) - sum(s_intersection_gold_set))

        return numerator / denominator

    def score_tuple(self, gold_tuple: [], system_tuple: []) -> float:
        # Both tuples should be of the same length.
        # If system tuple is empty, match its length.
        if len(system_tuple) == 1 and system_tuple[0] == '':
            system_tuple = ['', '', '', '']
        assert len(gold_tuple) == len(system_tuple) == 4
        gold_step_id = gold_tuple[3] if len(gold_tuple) == 4 else ""
        system_step_id = system_tuple[3] if len(system_tuple) == 4 else ""
        # as a precondition of match, the predicted step_id must match.
        if gold_step_id != system_step_id:
            return 0.0
        # sum over different components.
        score = 0.0
        for i in range(0, len(gold_tuple) - 1):
            score += self.match_score(gold_tuple[i], system_tuple[i])

        # Ignore event_id match as it is a precondition
        return score / (1.0 * (len(gold_tuple) - 1))

    def score_answer_elem(self, gold_answer: [], system_answer: []) -> float:
        # Question 3 and 4 that involve tuples are scored differently.
        # Tuple gold answers will always contain step ids (precondition to match against gold).
        is_answer_a_tuple = len(gold_answer) > 1 and len(gold_answer[3]) > 0

        if is_answer_a_tuple:
            # answer example: [init_value, final_value, loc_value, step_value]
            # e.g., (plants, ?, sediment, event2)
            return self.score_tuple(gold_answer, system_answer)
        else:
            # answer example: "water OR dew" or other example: "water"
            # As these are tuples of length 1 or tuples with no step ids,
            # match the first column of the tuples.
            return self.match_score(gold_answer[0], system_answer[0])

    # def score_question_containing_no_tuple(self, gold_answers: [[]], system_answers: [[]]) -> Metrics:
    #     # One metric per question.
    #     metrics = Metrics()
    #     total_score = 0.0
    #
    #     # total_score reflects (a intersection b) score
    #     for g, s in zip(gold_answers, system_answers):
    #         total_score += self.score_answer_elem(g, s)
    #
    #     # Ensure that the values of precision and recall <= 1.
    #     assert total_score <= max(len(system_answers), len(gold_answers))
    #     # edge case when system_answers = empty (then total_score =0).
    #     metrics.set_precision(0.0 if total_score == 0.0 else total_score / (1.0 * len(system_answers)))
    #     metrics.set_recall(0.0 if total_score == 0.0 else total_score / (1.0 * len(gold_answers)))
    #
    #     return metrics

    def is_empty(self, gold_answers):
        return len(gold_answers) == 0 or len(gold_answers[0][0]) == 0

        # Example 1: [g1, g2] [s1]
        # precision_numerator = max(g1s1,  g2s1)
        # recall_numerator    = max(g1s1) + max(g2s1)
        #
        # Example 2: [g1,g2] [s1,s2]
        # precision_numerator = max (g1s1, g2s1) + max (g1s2, g2s2)
        # recall_numerator    = max (g1s1, g2s1) + max (g1s2, g2s2)
        #
        # Example 3: [g1,g2] [s1,s2,s3]
        # precision_numerator = max (g1s1, g2s1) + max(g1s2, g2s2) + max (g1s3, g2s3)
        # recall_numerator    = max (g1s1, g1s2, g1s3) + max (g2s1, g2s2, g2s3)
        #
        # precision = precision_numerator / len(system_answers)
        # recall = recall_numerator / len(gold_answers)
    def score_question(self, gold_answers: [[]], system_answers: [[]]) -> Metrics:
        # One metric per question.
        metrics = Metrics()

        if self.is_empty(gold_answers) and self.is_empty(system_answers):
            metrics.set_recall(1.0)
            metrics.set_precision(1.0)
            return metrics
        if self.is_empty(gold_answers):
            metrics.set_recall(1.0)
            metrics.set_precision(0.0)
            return metrics
        if self.is_empty(system_answers):
            metrics.set_recall(0.0)
            metrics.set_precision(1.0)
            return metrics

        # Example 1: [g1, g2] [s1]
        # precision_numerator = max(g1s1,  g2s1)
        precision_numerator = 0.0
        for s in system_answers:
            curr_scores = []
            for g in gold_answers:
                curr_scores.append(self.score_answer_elem(g, s))
            precision_numerator += (max(curr_scores) if len(gold_answers) > 0 else 0.0)

        # recall_numerator = max(g1s1) + max(g2s1)
        # only compute this when len(S) != len(G)
        recall_numerator = precision_numerator
        if len(system_answers) != len(gold_answers):
            recall_numerator = 0.0
            for g in gold_answers:
                curr_scores = []
                for s in system_answers:
                    curr_scores.append(self.score_answer_elem(g, s))
                recall_numerator += (max(curr_scores) if len(system_answers) > 0 else 0.0)

        # edge case when system_answers = empty (then numerator =0).
        metrics.set_precision(0.0 if precision_numerator == 0.0 else precision_numerator / (1.0 * len(system_answers)))
        metrics.set_recall(0.0 if recall_numerator == 0.0 else recall_numerator / (1.0 * len(gold_answers)))

        return metrics

    def score_all_questions(self) -> dict:
        metrics_all = dict()  # process_id => question id => metrics
        for process_id in self.systems.keys():
            if process_id not in metrics_all.keys():
                metrics_all.setdefault(process_id, dict())
                # This should never really happen, so it is okay to throw an error.
                if process_id not in self.golds:
                    raise Exception("Evaluator: INFO: Possibly a problem with the data "
                                    "because the gold does not contain process id: ", process_id)

                # There are four questions in this task.
                for ques_id in [1, 2, 3, 4]:
                    metrics = self.score_question(
                        self.golds[process_id][ques_id],
                        self.systems[process_id][ques_id]
                    )
                    metrics_all[process_id][ques_id] = metrics
        return metrics_all

    # def ques_wise_overall_f1(self, metrics_all: dict):
    #     queswise = [0.0] * 4
    #     total_ques = 0
    #     for m in metrics_all.keys():
    #         all_ques = metrics_all[m]
    #         total_ques += 1
    #         for qid in all_ques.keys():
    #             # F1 is stored at index=3
    #             curr_f1 = all_ques[qid].get_scores()[3]
    #             queswise[qid - 1] += curr_f1
    #     # Macro average F1 score.
    #     returned = [round(x / (1.0 * total_ques), 3) for x in queswise]
    #     return returned

    def ques_wise_overall_metric(self, metrics_all: dict, metric_id=3):
        queswise = [0.0] * 4
        total_ques = 0
        for m in metrics_all.keys():
            all_ques = metrics_all[m]
            total_ques += 1
            for qid in all_ques.keys():
                # F1 is stored at index=3
                curr_f1 = all_ques[qid].get_scores()[metric_id]
                queswise[qid - 1] += curr_f1
        # Macro average F1 score.
        returned = [round(x / (1.0 * total_ques), 3) for x in queswise]
        return returned

    def pretty_print(self, metrics_all: dict):
        to_print = []
        for m in metrics_all.keys():
            to_print_dict = metrics_all[m]
            for qid in to_print_dict.keys():
                one_metric = to_print_dict[qid]
                to_print.append(str(m) + "\t" + str(qid) + "\t" + str(one_metric.get_scores()[3]))
        return '\n'.join(to_print)
