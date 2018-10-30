import enum
import os

import spacy
from allennlp.common.file_utils import cached_path

from propara.evaluation.eval import Evaluator

# Looking up lexicon with stem and roots.
nlp = spacy.load('en_core_web_sm')


class AI2LexiconPredicate(enum.Enum):
    IS_AT = 0
    NOT_IS_AT = 1
    EXISTS = 2
    NOT_EXISTS = 3
    HAS_TEMPERATURE = 4
    HAS_PHASE = 5
    HAS_SIZE = 6


class AI2LexiconPattern(enum.Enum):
    SO = 0
    S = 1
    O = 2


class AI2LexiconArg(enum.Enum):
    SUBJECT = 0
    OBJECT = 1
    PREP_SRC = 2
    PREP_DEST = 3
    INCREASED = 4
    DECREASED = 5
    SOLID = 6
    LIQUID = 7
    GAS = 8
    NONE = 9


class AI2LexiconIndications(enum.Enum):
    CREATED = 2
    DESTROYED = 1
    MOVED = 0
    PHASE_SOLID_LIQUID = 7
    PHASE_LIQUID_SOLID = 8
    PHASE_LIQUID_GAS = 9
    PHASE_GAS_LIQUID = 10
    PHASE_SOLID_GAS = 11
    PHASE_GAS_SOLID = 12
    PHASE_UNK_SOLID = 13  # Unknown to solid e.g. solidifies.
    PHASE_UNK_LIQUID = 14
    PHASE_UNK_GAS = 15
    TEMPERATURE_INC = 5
    TEMPERATURE_DEC = 6
    SIZE_INC = 3
    SIZE_DEC = 4
    UNK = 16


class AI2Lexicon:
    def __init__(self, lexicon_fp="tests/fixtures/ie/TheSemanticLexicon-v3.0_withadj.tsv"):
        """
        Loads the AI2 semantic lexicon.
        To query: before, after, and optionally pattern.
        :param lexicon_fp: tests/fixtures/ie/TheSemanticLexicon-v3.0_withadj.tsv
        """
        # verb => information.
        # absorb
        #   SUBJECT VERB OBJECT PREP-SRC PREP-DEST
        #   is-at(OBJECT, PREP-SRC), not is-at(OBJECT, SUBJECT),
        #   not is-at(OBJECT, PREP-SRC), is-at(OBJECT, SUBJECT),
        # another example of before/after
        #   not exists(SUBJECT), exists(PREP-SRC),
        #   exists(SUBJECT), not exists(PREP-SRC),
        print(f"Loading semantic lexicon...")
        self.lexicon = dict()
        self.original_entries = dict()
        self.load(lexicon_fp)
        self.lexicon_verbs_only = set()
        for v in [k[0] for k in self.lexicon.keys()]:
            self.lexicon_verbs_only.add(v)
            self.lexicon_verbs_only.add(Evaluator.stem(v))
        print(f"[done]")

    @classmethod
    def type_of_pattern_given_pattern(cls, has_subj, has_obj):
        has_both = has_subj and has_obj
        return AI2LexiconPattern.SO if has_both \
            else AI2LexiconPattern.S if has_subj \
            else AI2LexiconPattern.O

    @classmethod
    def type_of_pattern(cls, pattern):
        # SUBJECT VERB OBJECT PREP-SRC PREP-DEST
        has_subj = "SUBJECT" in pattern
        has_obj = "OBJECT" in pattern
        return AI2Lexicon.type_of_pattern_given_pattern(has_subj, has_obj)

    @classmethod
    def generate_key(cls, verb, has_agent, has_patient):
        # SUBJECT VERB OBJECT PREP-SRC PREP-DEST
        return verb, AI2Lexicon.type_of_pattern_given_pattern(has_subj=has_agent, has_obj=has_patient)

    @classmethod
    def generate_key_stemmed(cls, verb, has_agent, has_patient):
        # SUBJECT VERB OBJECT PREP-SRC PREP-DEST
        verb = nlp.vocab.morphology.lemmatizer(verb, 'verb')[-1]
        # verb = Evaluator.stem(verb)
        key1 = AI2Lexicon.generate_key(verb, has_agent, has_patient)
        if key1:  # being burned => burned (won't be detected until now)
            verb = nlp.vocab.morphology.lemmatizer(AI2Lexicon.drop_leading_articles_and_stopwords(verb), 'verb')[-1]
            key1 = AI2Lexicon.generate_key(
                verb,
                has_agent,
                has_patient
            )
        return key1

    # returns verb form, and whether it is in lexicon
    def verb_or_stem_in_lexicon(self, verb, pattern=AI2LexiconPattern.SO):
        verb_and_pattern = (verb, pattern)
        if not verb_and_pattern:
            return ""
        stemmed_verb = verb_and_pattern[0]
        if verb_and_pattern not in self.lexicon:
            stemmed_verb = Evaluator.stem(verb_and_pattern[0])
        if (stemmed_verb, verb_and_pattern[1]) in self.lexicon:
            return stemmed_verb
        else:  # Verb absent in lexicon.
            return ""

    def entire_entry_of(self, key):
        if not key:
            return []
        return self.lexicon.get(key, dict())

    def generate_best_key(self, verb, has_agent=True, has_patient=True):
        if not verb:
            return []
        key = AI2Lexicon.generate_key(verb, has_agent, has_patient)
        return key if key in self.lexicon \
            else AI2Lexicon.generate_key_stemmed(verb, has_agent, has_patient)

        # your other hand => hand
        # the hand => hand

    @staticmethod
    def drop_leading_articles_and_stopwords(p):
        # other and another can only appear after the primary articles in first line.
        articles = ["a ", "an ", "the ", "your ", "his ", "their ", "my ", "this ", "that ",
                    "another ", "other ", "more ", "less "]
        for article in articles:
            if p.lower().startswith(article):
                p = p[len(article):]
        words = p.split(" ")
        answer = ""
        for idx, w in enumerate(words):
            if AI2Lexicon.nlp.vocab[w].is_stop:
                continue
            else:
                answer = " ".join(words[idx:])
                break
        return answer

    nlp = spacy.load('en_core_web_sm')

    def contains_verb(self, verb):
        if not verb:
            return ""
        verb = verb.lower()
        if verb in self.lexicon_verbs_only:
            return verb
        lemma_verb = nlp.vocab.morphology.lemmatizer(verb, 'verb')[-1]
        if lemma_verb in self.lexicon_verbs_only:
            return lemma_verb
        stem_verb = Evaluator.stem(verb) in self.lexicon_verbs_only
        if stem_verb in self.lexicon_verbs_only:
            return stem_verb
        head_lemma = self.nlp.vocab.morphology.lemmatizer(
            AI2Lexicon.drop_leading_articles_and_stopwords(verb), 'verb')[-1]
        if head_lemma in self.lexicon_verbs_only:
            return head_lemma
        head_stem = AI2Lexicon.drop_leading_articles_and_stopwords(Evaluator.stem(verb)) in self.lexicon_verbs_only
        if head_stem in self.lexicon_verbs_only:
            return head_stem
        return ""

    # before_after: can be "after" or "before"
    def _entry(self, key, before_after: str):
        info = self.entire_entry_of(key)
        # example of spo: is-at(OBJECT, PREP-SRC)
        spos = info.get(before_after, [])
        return spos

    # spos: are all tuples is-at, OBJECT, PREP-SRC , etc.
    # str_at_pos1: OBJECT in the example above.
    # Returns a dict of AI2LexiconPredicate and optional value.
    def _inner_entries(self, spos, str_at_pos1=None):
        return {spo[0]: (spo[2] or AI2LexiconArg.NONE) for spo in spos
                if spo and len(spo) >= 2 and (not str_at_pos1 or spo[1] == str_at_pos1)}

    def _after_subj(self, key):
        return self._inner_entries(self._entry(key, "after"), "SUBJECT")

    def _after_obj(self, key):
        return self._inner_entries(self._entry(key, "after"), "OBJECT")

    def _before_subj(self, key):
        return self._inner_entries(self._entry(key, "before"), "SUBJECT")

    def _before_obj(self, key):
        return self._inner_entries(self._entry(key, "before"), "OBJECT")

    def what_happens_to_subj(self, verb, has_agent=True, has_patient=True):
        if not verb:
            return []
        key = self.generate_best_key(verb, has_agent, has_patient)
        before = self._before_subj(key)
        after = self._after_subj(key)
        return self._what_happens_to_(before, after)

    def what_happens_to_obj(self, verb, has_agent=True, has_patient=True):
        if not verb:
            return []
        key = self.generate_best_key(verb, has_agent, has_patient)
        before = self._before_obj(key)
        after = self._after_obj(key)
        return self._what_happens_to_(before, after)

    def _positive_indications(self, what_happens_to_dict):
        return [k for k, v in what_happens_to_dict.items() if v]

    def _what_happens_to_(self, before, after):
        # sample_before_or_after: {
        #     AI2LexiconPredicate.IS_AT: AI2LexiconArg.OBJECT,
        #     AI2LexiconPredicate.NOT_IS_AT: AI2LexiconArg.PREP_SRC,
        # }
        return self._positive_indications({
            AI2LexiconIndications.CREATED: self._creation_indication(before, after),
            AI2LexiconIndications.DESTROYED: self._destruction_indication(before, after),
            AI2LexiconIndications.MOVED: self._movement_indication(before, after),
            AI2LexiconIndications.TEMPERATURE_INC: self._temperature_inc_indication(before, after),
            AI2LexiconIndications.TEMPERATURE_DEC: self._temperature_dec_indication(before, after),
            AI2LexiconIndications.SIZE_INC: self._size_inc_indication(before, after),
            AI2LexiconIndications.SIZE_DEC: self._size_dec_indication(before, after),
            AI2LexiconIndications.PHASE_GAS_LIQUID: self._phase_gas_liquid_indication(before, after),
            AI2LexiconIndications.PHASE_LIQUID_GAS: self._phase_liquid_gas_indication(before, after),
            AI2LexiconIndications.PHASE_SOLID_LIQUID: self._phase_solid_liquid_indication(before, after),
            AI2LexiconIndications.PHASE_LIQUID_SOLID: self._phase_liquid_solid_indication(before, after),
            AI2LexiconIndications.PHASE_GAS_SOLID: self._phase_gas_solid_indication(before, after),
            AI2LexiconIndications.PHASE_SOLID_GAS: self._phase_solid_gas_indication(before, after),
            AI2LexiconIndications.PHASE_UNK_SOLID: self._phase_unk_solid_indication(before, after),
            AI2LexiconIndications.PHASE_UNK_LIQUID: self._phase_unk_liquid_indication(before, after),
            AI2LexiconIndications.PHASE_UNK_GAS: self._phase_unk_gas_indication(before, after)
        })

    def _size_inc_indication(self, before, after):
        return AI2LexiconPredicate.HAS_SIZE in after and after[
                                                             AI2LexiconPredicate.HAS_SIZE] == AI2LexiconArg.INCREASED

    def _size_dec_indication(self, before, after):
        return AI2LexiconPredicate.HAS_SIZE in after and after[
                                                             AI2LexiconPredicate.HAS_SIZE] == AI2LexiconArg.DECREASED

    def _temperature_dec_indication(self, before, after):
        return AI2LexiconPredicate.HAS_TEMPERATURE in after and after[
                                                                    AI2LexiconPredicate.HAS_TEMPERATURE] == AI2LexiconArg.DECREASED

    def _temperature_inc_indication(self, before, after):
        return AI2LexiconPredicate.HAS_TEMPERATURE in after and after[
                                                                    AI2LexiconPredicate.HAS_TEMPERATURE] == AI2LexiconArg.INCREASED

    def _phase_liquid_solid_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before and before[
                                                               AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.LIQUID and AI2LexiconPredicate.HAS_PHASE in after and \
               after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.SOLID

    def _phase_solid_liquid_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before and before[
                                                               AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.SOLID and AI2LexiconPredicate.HAS_PHASE in after and \
               after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.LIQUID

    def _phase_solid_gas_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before and before[
                                                               AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.SOLID and AI2LexiconPredicate.HAS_PHASE in after and \
               after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.GAS

    def _phase_gas_solid_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before and before[
                                                               AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.GAS and AI2LexiconPredicate.HAS_PHASE in after and \
               after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.LIQUID

    def _phase_gas_liquid_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before and before[
                                                               AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.GAS and AI2LexiconPredicate.HAS_PHASE in after and \
               after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.LIQUID

    def _phase_liquid_gas_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before and before[
                                                               AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.LIQUID and AI2LexiconPredicate.HAS_PHASE in after and \
               after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.GAS

    def _phase_unk_solid_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before \
               and before[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.NONE \
               and AI2LexiconPredicate.HAS_PHASE in after \
               and after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.SOLID

    def _phase_unk_liquid_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before \
               and before[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.NONE \
               and AI2LexiconPredicate.HAS_PHASE in after \
               and after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.LIQUID

    def _phase_unk_gas_indication(self, before, after):
        return AI2LexiconPredicate.HAS_PHASE in before \
               and before[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.NONE \
               and AI2LexiconPredicate.HAS_PHASE in after \
               and after[AI2LexiconPredicate.HAS_PHASE] == AI2LexiconArg.GAS

    def _destruction_indication(self, before, after):
        # exists and then not exists
        return AI2LexiconPredicate.EXISTS in before and AI2LexiconPredicate.NOT_EXISTS in after

    def _creation_indication(self, before, after):
        # not exists and then exists
        return AI2LexiconPredicate.NOT_EXISTS in before and AI2LexiconPredicate.EXISTS in after

    def _movement_indication(self, before, after):
        # init loc != final loc
        return before.get(AI2LexiconPredicate.IS_AT, AI2LexiconArg.NONE) != \
               after.get(AI2LexiconPredicate.IS_AT, AI2LexiconArg.NONE)

    # Location is different and can be drawn from an open vocab.
    # This function provides init and final loc according to the lexicon.
    # e.g., x absorbs y, y moves to final_loc=x
    def movement_before_after(self, key, is_subj):
        before_loc = ""
        after_loc = ""
        before = self._before_subj(key) if is_subj else self._before_obj(key)
        after = self._after_subj(key) if is_subj else self._after_obj(key)
        if self._movement_indication(before, after):
            before_loc = before.get(AI2LexiconPredicate.IS_AT, AI2LexiconArg.NONE)
            after_loc = after.get(AI2LexiconPredicate.IS_AT, AI2LexiconArg.NONE)
        return before_loc, after_loc

    @staticmethod
    def revise_sem_lexicon_onetime(adj_lexicon_fp, orig_lexicon_fp, updated_lexicon_fp):
        print(f"Adding grounded adjective patterns to : {orig_lexicon_fp} \n\tusing adjectives : {adj_lexicon_fp}")
        is_first_line = True
        # Hardcoded inflections for a handful of adjective paired verbs such as "becomes", simplifying lookups.
        # Modals such as "can become" ADJ are ignored because only the substring "become ADJ" is of interest.
        inflections = {
            "become": "become, became, becoming, becomes",
            "turn into": "turn into, turned into, turning into, turns into",
            "turn": "turn, turned, turning, turns",
            "change to": "change to, changed to, changing to, changes to",
            "get": "get, got, getting, gets"
        }
        # adjective related entries to append to semantic lexicon.
        new_entries = []
        with open(adj_lexicon_fp, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#') or is_first_line:
                    is_first_line = False
                    continue
                cols = [str.strip(c) for c in line.split("\t")]
                if len(cols) < 11:
                    continue
                # cols[0] = "become, turn into, turn, change to, get"
                keys = []
                adj = cols[2]
                source = f"ADJ:{cols[-1]}"
                for v in cols[0].split(","):
                    v = str.strip(v)
                    # fetch inflections e.g., become => become, became, becomes, becoming, becomes.
                    for v_inflections in [str.strip(t) for t in inflections.get(v, v).split(",")]:
                        keys.append(f"{v_inflections} {adj}")
                before_temp = cols[3]
                after_temp = cols[4]
                before_phase = cols[5]
                after_phase = cols[6]
                before_size = cols[7]
                after_size = cols[8]
                after_part = ""
                before_part = ""
                if after_temp:
                    after_part += f"has-temperature(SUBJECT, {after_temp}), "
                    before_part += f"has-temperature(SUBJECT, {before_temp}), "
                if after_phase:
                    after_part += f"has-phase(SUBJECT, {after_phase}), "
                    before_part += f"has-phase(SUBJECT, {before_phase}), "
                if after_size:
                    after_part += f"has-size(SUBJECT, {after_size}), "
                    before_part += f"has-size(SUBJECT, {before_size}), "

                pattern = cols[1]
                for key in keys:
                    new_entries.append(f"{source}\t{key}\t{pattern}\t{before_part}\t{after_part}")
        # Add new entries.
        outfile = open(updated_lexicon_fp, 'w')
        with open(orig_lexicon_fp, 'r') as infile:
            for line in infile:
                # Remove lines dependent on adjective patterns
                # we next add all grounded adjective patterns.
                if "ADJ" not in line and "\tturn\t" not in line:  # also, remove the very ambiguous "turn"
                    outfile.write(f"{line.strip()}\n")
        for new_entry in new_entries:
            outfile.write(f"{new_entry}\n")
        print(f"New semantic lexicon in: {updated_lexicon_fp}")
        outfile.close()

    def get_original_entry(self, v):
        return self.original_entries.get(v, f"{v} not in original lexicon")

    def load(self, from_file_path):
        is_first_line = True
        print(f"Loading semantic lexicon from {from_file_path}, path exists? "
              f"{os.path.exists(from_file_path)}")
        if not os.path.exists(from_file_path):
            print(f"Trying to load from alternative S3 bucket.")
            from_file_path = cached_path("https://s3-us-west-2.amazonaws.com/ai2-aristo-propara/dataset"
                                         "/TheSemanticLexicon-v3.0_withadj.tsv")
        with open(from_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#') or is_first_line:
                    is_first_line = False
                    continue
                # cols: AI2=[0], stop=[1], SUBJECT VERB PREP-SRC PREP-DEST=[2],
                # is-at(SUBJECT, PREP-SRC), not is-at(SUBJECT, PREP-DEST)=[3]
                # not is-at(SUBJECT, PREP-SRC), is-at(SUBJECT, PREP-DEST)=[4]
                cols = line.split("\t")
                # Ignore unannotated entries (no before/after).
                # Lexicon entries contain either both or no before and after.
                if len(cols) < 4:
                    continue
                pattern = AI2Lexicon.type_of_pattern(cols[2])
                verbs = set()
                orig_verb = cols[1]
                if orig_verb not in self.original_entries:
                    self.original_entries.setdefault(orig_verb, [])
                self.original_entries[orig_verb].append(line)

                verbs.add(orig_verb)
                lemma_verb = nlp.vocab.morphology.lemmatizer(orig_verb, 'verb')[-1]
                stemmed_verb = Evaluator.stem(orig_verb)
                # root_verb = AI2Lexicon.drop_leading_articles_and_stopwords(Evaluator.stem(verb))
                verbs.add(orig_verb)
                if len(lemma_verb) > 0:
                    verbs.add(lemma_verb)
                if stemmed_verb != lemma_verb:
                    verbs.add   (stemmed_verb)
                # Add multiple entries in lexicon (otherwise, buried becomes buri).
                for verb in verbs:
                    key = verb, pattern
                    if key not in self.lexicon.keys():
                        self.lexicon.setdefault(key, dict())
                    # Skipping source and pattern.
                    self.load_before_or_after("before", cols, key)
                    self.load_before_or_after("after", cols, key)

    def load_before_or_after(self, before_or_after: str, cols, key):
        idx_to_load = 3 if before_or_after == "before" else 4
        befores = [b for b in cols[idx_to_load].strip().split("),") if b]
        self.lexicon[key].setdefault(before_or_after, [])
        for b in befores:
            # not exists(PREP-DEST)
            # b_p = not exists
            #       must be converted to AI2LexiconPredicate.NOT_EXISTS
            b_splits = b.split('(')
            b_p_str = b_splits[0].strip().replace(' ', '_').replace('-', '_').upper()
            b_p = AI2LexiconPredicate[b_p_str]
            b_so = b_splits[1].split(',')
            b_s = b_so[0].strip()
            b_o_input = b_so[1].strip().replace('(', '').replace(')', '').replace(' ', '_').replace('-',
                                                                                                    '_').upper() if len(
                b_so) > 1 else ""
            b_o_input = b_o_input.replace("SOILD", "SOLID")  # typo in v3.0
            b_o = AI2LexiconArg[b_o_input] if b_o_input else AI2LexiconArg.NONE
            self.lexicon[key][before_or_after].append((b_p, b_s, b_o))
