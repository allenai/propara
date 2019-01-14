from propara.evaluation.eval import Evaluator


class Parti2ID:

    def __init__(self,
                 casing:bool=True,
                 stemming:bool=True,
                 separator:bool=True):
        self.configurations_casing = casing
        self.configurations_separator = separator
        self.configurations_stemming_and_no_leading_articles = stemming
        self.literal_match_type = "match_type"
        self.literal_matched_parti = "literal_matched_parti"
        self.literal_matched_parti_str = "literal_matched_parti_str"
        self.literal_matched_direct = "direct"
        self.literal_matched_casing = "casing"
        self.literal_matched_stemming = "stemming_and_no_leading_articles"

    def best_matching_id(self, q:str, p_map:dict):
        """
        :param q: query e.g., leaf 
        :param p_map: [animals: 1, leaves: 2 , plant: 3]
        :return: id: in this example, e.g.,
            return {"match_type": "stemming_and_no_leading_articles", "matched":2} 
        """
        q = str.strip(q)
        match_direct = [(t, v) for t, v in p_map.items()
                        if t == q]
        if match_direct:
            return {self.literal_match_type: self.literal_matched_direct,
                    self.literal_matched_parti: match_direct[0][1],
                    self.literal_matched_parti_str: match_direct[0][0]}

        match_casing = [(t, v) for t, v in p_map.items()
                        if str.lower(t) == str.lower(q)]
        if match_casing and self.configurations_casing:
            return {self.literal_match_type: self.literal_matched_casing,
                    self.literal_matched_parti: match_casing[0][1],
                    self.literal_matched_parti_str: match_casing[0][0]}

        q_stemmed = Evaluator.stem(q)
        match_stemming_and_no_leading_articles = [(t, v) for t, v in p_map.items()
                                                  if Evaluator.stem(t) == q_stemmed]
        if match_stemming_and_no_leading_articles and self.configurations_stemming_and_no_leading_articles:
            # print(f"match_stemming_and_no_leading_articles:{match_stemming_and_no_leading_articles}")
            return {self.literal_match_type: self.literal_matched_stemming,
                    self.literal_matched_parti: match_stemming_and_no_leading_articles[0][1],
                    self.literal_matched_parti_str: match_stemming_and_no_leading_articles[0][0]}

        return {}

    def best_matching_id_with_separator(self,
                                        q:str,
                                        p_map:dict,
                                        potential_separators = set([" and ", " or ", ";", ","])
                                        ):
        """
        :param q: tree, plant
        :param p_map: keys are not separated {trees: 1, plants: 1, ...}
        :return: 
        """
        assert self.configurations_separator, "This function (best_matching_id_with_separator)" \
                                              "should not be invoked if configurations_split_on_separator " \
                                              "is False."
        q_lower = str.lower(q)  # to allow matching " and " ; " or "
        for sep in potential_separators:
            if sep in q_lower:

                # loop over the splitted q.
                for q_sep in q.split(sep):
                    matched = self.best_matching_id(q=q_sep, p_map=p_map)
                    if matched:
                        return {self.literal_match_type: matched[self.literal_match_type]+"_separator_"+sep,
                                self.literal_matched_parti: matched[self.literal_matched_parti],
                                self.literal_matched_parti_str: matched[self.literal_matched_parti_str]}

        return {}

