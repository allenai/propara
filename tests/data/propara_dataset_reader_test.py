# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField, ListField, SequenceLabelField, LabelField, SpanField
from allennlp.data.tokenizers.token import Token
import pytest

from propara.data.propara_dataset_reader import ProParaDatasetReader, _find_span

FILENAME = "tests/fixtures/state_changes/grids.small.tsv"

class TestProParaDatasetReader(AllenNlpTestCase):
    def test_find_span(self):
        sentence = [Token("My"), Token("car"), Token("is"), Token("-"), Token("grey"), Token("?")]

        # Single token
        assert _find_span([Token("car")], sentence) == (1, 1)

        # Multi token
        assert _find_span([Token("My"), Token("car")], sentence) == (0, 1)

        # Case insensitive
        assert _find_span([Token("my"), Token("car")], sentence) == (0, 1)

        # Not in sentence
        assert _find_span([Token("my"), Token("truck")], sentence) == (-1, -1)

        # Unknown
        assert _find_span([Token("?")], sentence) == (-2, -2)

        # Absent
        assert _find_span([Token("-")], sentence) == (-3, -3)


    def test_read_from_file(self):
        reader = ProParaDatasetReader()

        instances = list(reader.read(FILENAME))

        instance = instances[0]
        fields = instance.fields

        # 4	SID	PARTICIPANTS	plants	bacteria	sediment	oil
        # 4		PROMPT: How does oil form?	-=====	-=====	-=====	-=====
        # 4	state1		?	?	?	-
        # 4	event1	Plants die.
        # 4	state2		?	?	?	-
        # 4	event2	They are buried in sediment.
        # 4	state3		sediment	?	?	-
        # 4	event3	Bacteria is buried in the sediment.
        # 4	state4		sediment	sediment	?	-
        # 4	event4	Large amounts of sediment gradually pile on top of the original sediment.
        # 4	state5		sediment	sediment	?	-
        # 4	event5	Pressure builds up.
        # 4	state6		sediment	sediment	?	-
        # 4	event6	Heat increases.
        # 4	state7		sediment	sediment	?	-
        # 4	event7	The chemical structure of the buried sediment and plants changes.
        # 4	state8		sediment	sediment	?	-
        # 4	event8	The sediment and plants are at least one mile underground.
        # 4	state9		one mile underground	sediment	underground	-
        # 4	event9	The buried area is extremely hot.
        # 4	state10		one mile underground	sediment	underground	-
        # 4	event10	More chemical changes happen eand the buried material becomes oil.
        # 4	state11		-	-	underground	underground

        participants = fields["participants"]
        assert isinstance(participants, ListField)

        pfl = participants.field_list
        assert len(pfl) == 4
        assert all(isinstance(field, TextField) for field in pfl)
        assert all(len(field.tokens) == 1 for field in pfl)
        assert {field.tokens[0].text for field in pfl} == {'plants', 'bacteria', 'sediment', 'oil'}

        participant_strings = fields["participant_strings"].metadata
        assert participant_strings == ['plants', 'bacteria', 'sediment', 'oil']

        sentences = fields["sentences"]
        assert isinstance(sentences, ListField)

        sfl = sentences.field_list
        assert len(sfl) == 10
        assert all(isinstance(field, TextField) for field in sfl)
        sentence = sfl[0].tokens
        assert [token.text for token in sentence] == ["Plants", "die", "."]


        verbs = fields["verbs"]
        assert isinstance(verbs, ListField)

        vfl = verbs.field_list
        assert len(vfl) == 10
        assert all(isinstance(field, SequenceLabelField) for field in vfl)
        # second word is the verb
        assert vfl[0].labels == [0, 1, 0]


        actions = fields["actions"]
        assert isinstance(actions, ListField)

        afl = actions.field_list
        assert len(afl) == 4  # one per participant
        assert all(isinstance(af, ListField) for af in afl)
        assert all(len(af.field_list) == 10 for af in afl)
        af0 = afl[0]
        assert all(isinstance(action, LabelField) for action in af0.field_list)
        assert [action.label for action in af0.field_list] == [0, 3, 0, 0, 0, 0, 0, 3, 0, 2]


        starts = fields["before_locations"]
        assert isinstance(starts, ListField)

        sfl = starts.field_list
        assert len(sfl) == 4  # one per participant
        assert all(isinstance(sf, ListField) for sf in sfl)
        assert all(len(sf.field_list) == 10 for sf in sfl)
        sf0 = sfl[0]
        assert all(isinstance(span, SpanField) for span in sf0.field_list)
        assert ([(span.span_start, span.span_end) for span in sf0.field_list] ==
                [(-2, -2), (-2, -2), (5, 5), (3, 3), (-1, -1), (-1, -1), (6, 6), (1, 1), (-1, -1), (-1, -1)])


        ends = fields["after_locations"]
        assert isinstance(ends, ListField)

        efl = ends.field_list
        assert len(efl) == 4  # one per participant
        assert all(isinstance(ef, ListField) for ef in efl)
        assert all(len(ef.field_list) == 10 for ef in efl)
        ef0 = efl[0]
        assert all(isinstance(span, SpanField) for span in ef0.field_list)
        assert ([(span.span_start, span.span_end) for span in ef0.field_list] ==
                [(-2, -2), (4, 4), (5, 5), (3, 3), (-1, -1), (-1, -1), (6, 6), (7, 9), (-1, -1), (-3, -3)])

        participant_indicators = fields["participant_indicators"]
        # should be (num_participants, num_sentences, num_words)
        pifl = participant_indicators.field_list
        assert len(pifl) == 4  # one per participant
        assert all(isinstance(pif, ListField) for pif in pifl)
        assert all(len(pif.field_list) == 10 for pif in pifl)  # 10 sentences
        pif0 = pifl[0].field_list
        assert all(isinstance(pif, SequenceLabelField) for pif in pif0)
        # plants -> Plants die.
        assert pif0[0].labels == [1, 0, 0]
        # plants -> They are buried in sediment.
        assert pif0[1].labels == [0, 0, 0, 0, 0, 0]
        # plants -> The sediment and plants are at least one mile underground.
        assert pif0[7].labels == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        # Paragraph indicators
        paragraph = fields["paragraph"]
        num_tokens = len(paragraph.tokens)
        assert num_tokens == sum(len(sf.tokens) for sf in sentences.field_list)

        # Paragraph verb indicators
        paragraph_verb_labels = fields["paragraph_verbs"].labels
        assert paragraph_verb_labels == [label for sentence_verbs in vfl for label in sentence_verbs.labels]

        # Paragraph participant indicators
        ppi = fields["paragraph_participant_indicators"]
        assert len(ppi.field_list) == len(participants.field_list)

        for i, participant_i_indicator in enumerate(ppi.field_list):
            joined_labels = [label for pif in pifl[i].field_list for label in pif.labels]
            assert joined_labels == participant_i_indicator.labels

        # Paragraph sentence indicators
        psi = fields["paragraph_sentence_indicators"]
        assert len(psi.field_list) == len(sentences.field_list)

        length0 = len(sentences.field_list[0].tokens)
        length1 = len(sentences.field_list[1].tokens)
        length2 = len(sentences.field_list[2].tokens)

        psi2 = psi.field_list[2]

        for i, label in enumerate(psi2.labels):
            if i < length0 + length1:
                assert label == 0
            elif i < length0 + length1 + length2:
                assert label == 1
            else:
                assert label == 2
