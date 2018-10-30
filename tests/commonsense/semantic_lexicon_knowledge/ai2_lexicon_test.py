from unittest import TestCase

from propara.commonsense.semantic_lexicon_knowledge.ai2_lexicon import AI2Lexicon, AI2LexiconPredicate, AI2LexiconArg, AI2LexiconIndications, \
    AI2LexiconPattern


class TestAI2Lexicon(TestCase):
    def setUp(self):
        self.lexicon_fp = "tests/fixtures/ie/TheSemanticLexicon-v3.0_withadj.tsv"

    def testLoads(self):
        self.lexicon = AI2Lexicon(self.lexicon_fp)
        # print(f"evaporate.subj: {self.lexicon.what_happens_to_subj('evaporate', has_agent=True, has_patient=False)}")
        # print(f"evaporate.obj: {self.lexicon.what_happens_to_obj('evaporate', has_agent=True, has_patient=False)}")
        #
        # print(f"evaporate.subj: {self.lexicon.what_happens_to_subj('evaporate')}")
        # print(f"evaporate.obj: {self.lexicon.what_happens_to_obj('evaporate')}")

        # v2 doesn't contain size, temperature, phase attributes
        # infile = "tests/fixtures/ie/ai2-lexicon-v2.tsv"
        # the following path is useful when debugging from browser.
        # self.lexicon = AI2Lexicon("tests/fixtures/ie/TheSemanticLexicon-v3.0_withadj.tsv")
        assert self.lexicon._after_subj(("blend in", AI2LexiconPattern.SO)) == {
            AI2LexiconPredicate.IS_AT: AI2LexiconArg.OBJECT,
            AI2LexiconPredicate.NOT_IS_AT: AI2LexiconArg.PREP_SRC,
        }
        assert self.lexicon._after_obj(("absorb", AI2LexiconPattern.SO))[
                   AI2LexiconPredicate.IS_AT] == AI2LexiconArg.SUBJECT
        # assert self.lexicon._after_obj(("absorbs", AI2LexiconPattern.SO)).get(AI2LexiconPredicate.IS_AT, "") == AI2LexiconArg.SUBJECT
        assert len(self.lexicon._after_obj(("blend in", AI2LexiconPattern.SO))) == 0
        assert len(self.lexicon._after_obj(("blend blend2", AI2LexiconPattern.SO))) == 0
        assert AI2LexiconIndications.MOVED not in self.lexicon.what_happens_to_subj("absorbs")
        assert AI2LexiconIndications.MOVED in self.lexicon.what_happens_to_obj("absorbs")
        assert AI2LexiconIndications.CREATED in self.lexicon.what_happens_to_obj("sprout")
        assert AI2LexiconIndications.CREATED in self.lexicon.what_happens_to_subj("sprout", has_agent=True,
                                                                                  has_patient=False)
        assert AI2LexiconIndications.DESTROYED not in self.lexicon.what_happens_to_subj("sprout")
        assert AI2LexiconIndications.DESTROYED not in self.lexicon.what_happens_to_obj("sprout")
        assert AI2LexiconIndications.TEMPERATURE_INC not in self.lexicon.what_happens_to_obj("turn")
        assert AI2LexiconIndications.TEMPERATURE_INC in self.lexicon.what_happens_to_subj("gets hot")
        assert AI2LexiconIndications.SIZE_INC in self.lexicon.what_happens_to_subj("gets bigger")
        assert AI2LexiconIndications.SIZE_INC in self.lexicon.what_happens_to_subj("become bigger")
        assert AI2LexiconIndications.SIZE_INC in self.lexicon.what_happens_to_subj("turned bigger")
        assert AI2LexiconIndications.SIZE_INC not in self.lexicon.what_happens_to_obj("turns into bigger")
        assert AI2LexiconIndications.MOVED not in self.lexicon.what_happens_to_subj("turned")
        assert AI2LexiconIndications.PHASE_UNK_GAS in self.lexicon.what_happens_to_subj("turned gaseous")
        assert AI2LexiconIndications.PHASE_LIQUID_SOLID in self.lexicon.what_happens_to_subj("solidify", has_agent=True,
                                                                                             has_patient=False)
        assert AI2LexiconIndications.PHASE_LIQUID_SOLID in self.lexicon.what_happens_to_obj("solidify", has_agent=True,
                                                                                            has_patient=True)
        assert AI2LexiconIndications.PHASE_UNK_SOLID not in self.lexicon.what_happens_to_subj("solidifies")
        assert AI2LexiconIndications.PHASE_SOLID_GAS in self.lexicon.what_happens_to_subj("sublime", has_agent=True,
                                                                                          has_patient=False)
        assert AI2LexiconIndications.PHASE_SOLID_GAS in self.lexicon.what_happens_to_obj("sublime", has_agent=True,
                                                                                         has_patient=True)

        # if agent and patient both are present or only 1
        # the difference is whether object is given or not
        # this happens for all verbs that can be both transitive/intransitive
        # they will have 2 entries.
        #
        # A big rock stops the stream of water from uphill => stream of water moved from uphill to rock
        # car stops at the intersection ==> car moved to intersection
        # we have removed lots of fine details in the patterns (VerbNet had much more info there)
        # if agent and patient both are present or only 1

    def test_type_of_pattern(self):
        input = "SUBJECT VERB OBJECT PREP-SRC PREP-DEST"
        assert AI2Lexicon.type_of_pattern(input) == AI2LexiconPattern.SO
        input = "SUBJECT VERB OBJECT"
        assert AI2Lexicon.type_of_pattern(input) == AI2LexiconPattern.SO
        input = "SUBJECT VERB PREP-SRC PREP-DEST"
        assert AI2Lexicon.type_of_pattern(input) == AI2LexiconPattern.S
