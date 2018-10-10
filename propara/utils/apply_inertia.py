#!/usr/bin/env python3

from typing import *
import contextlib
import sys
import json
from collections import namedtuple
import logging
import re

JsonDict = Dict[str, Any]

class OutputLine:
    def __init__(self, paragraph_id: int, sentence_id: int, entity: str):
        self.paragraph_id = paragraph_id
        self.sentence_id = sentence_id
        self.entity = entity
        self.before_location = "unk"
        self.after_location = "unk"

    def render(self) -> str:
        return "\t".join([
            str(self.paragraph_id),
            str(self.sentence_id),
            self.entity,
            self.before_location if self.before_location is not None else "null",
            self.after_location if self.after_location is not None else "null"
        ])

    def __str__(self):
        return "OutputLine(%r, %r, %r) at (%r, %r)" % (
            self.paragraph_id,
            self.sentence_id,
            self.entity,
            self.before_location,
            self.after_location
        )

Sentence = namedtuple("Sentence", [
    "sentence_id",
    "action",
    "from_location",
    "to_location"
])

def sentence_with_defaults(sentence_id: int) -> Sentence:
    return Sentence(sentence_id, 'NONE', None, None)

def text_from_file(filename: Optional[str]) -> Generator[str, None, None]:
    if filename is None:
        @contextlib.contextmanager
        def open_stdin(*args, **kwargs):
            yield sys.stdin
        open_fn = open_stdin
    elif filename.endswith(".bz2"):
        import bz2
        open_fn = bz2.open
    elif filename.endswith(".gz"):
        import gzip
        open_fn = gzip.open
    else:
        open_fn = open

    with open_fn(filename, "rt", encoding="UTF-8", errors="replace") as p:
        yield from p


def json_from_file(filename: Optional[str]) -> Generator[JsonDict, None, None]:
    for line in text_from_file(filename):
        try:
            yield json.loads(line)
        except ValueError as e:
            logging.warning("Error while reading document (%s); skipping", e)

@contextlib.contextmanager
def file_or_stdout(filename: Optional[str]):
    if filename is None:
        yield sys.stdout
    else:
        if filename.endswith(".gz"):
            import gzip
            open_fn = gzip.open
        elif filename.endswith(".bz2"):
            import bz2
            open_fn = bz2.open
        else:
            open_fn = open

        file = open_fn(filename, "wt", encoding="UTF-8")
        try:
            yield file
        finally:
            file.close()

def index_not_stupid(haystack, needle):
    try:
        return haystack.index(needle)
    except ValueError:
        return None

def apply_inertia(predictions_file: str, grid_file: str) -> Iterable[OutputLine]:
    # read the predictions file
    paragraph_ids_we_care_about = set()
    paragraphid_entity_to_sentences = {}
    for j in json_from_file(predictions_file):
        paragraph_id = int(j['paraid'])
        paragraph_ids_we_care_about.add(paragraph_id)
        sentence_id = int(j['sentenceid'])
        entity = j['entity']
        action = j['predicted_types']

        def string_with_tag(tag: str) -> Optional[str]:
            predicted_tags = [tag[2:] for tag in j['predicted_tags']]
            if tag in predicted_tags:
                begin_index = predicted_tags.index(tag)
                end_index = begin_index + 1
                while end_index < len(predicted_tags) and predicted_tags[end_index] == tag:
                    end_index += 1
                if tag in predicted_tags[end_index:]:
                    return None # There is another tagged sequence, so we ignore all of them.
                else:
                    return " ".join(j['sentence_tokens'][begin_index:end_index])
            else:
                return None

        to_location = string_with_tag("LOC-TO")
        from_location = string_with_tag("LOC-FROM")

        sentences = paragraphid_entity_to_sentences.setdefault((paragraph_id, entity), [])
        sentences.append(Sentence(sentence_id, action, from_location, to_location))

    # read the grid file
    paragraphid_to_sentenceids = {}
    paragraphid_participant = set()
    current_paragraph_id = None
    for line in text_from_file(grid_file):
        line = line.strip()
        if len(line) <= 0:
            current_paragraph_id = None
            continue
        line = re.split('[;\t]', line)
        if len(line) <= 0:
            current_paragraph_id = None
        elif current_paragraph_id != int(line[0]):
            current_paragraph_id = int(line[0])
            if current_paragraph_id in paragraph_ids_we_care_about:
                participants = frozenset(line[3:])
                paragraphid_participant |= {(current_paragraph_id, p) for p in participants}
        elif line[1].startswith("event") and current_paragraph_id in paragraph_ids_we_care_about:
            sentence_id = line[1]
            sentence_id = sentence_id[len("event"):]
            sentence_id = int(sentence_id)
            sentence_ids = paragraphid_to_sentenceids.setdefault(current_paragraph_id, set())
            sentence_ids.add(sentence_id)

    paragraphid_participant = list(paragraphid_participant)
    paragraphid_participant.sort()
    for paragraph_id, entity in paragraphid_participant:
        sentence_ids = paragraphid_to_sentenceids[paragraph_id]
        sentence_ids = list(range(1, max(sentence_ids) + 1))    # TODO: I suspect this does nothing
        result = {
            sentence_id : OutputLine(paragraph_id, sentence_id, entity)
            for sentence_id in sentence_ids
        }

        sentences = paragraphid_entity_to_sentences.get((paragraph_id, entity), [])
        sentenceid_to_sentence = {
            sentence.sentence_id : sentence
            for sentence in sentences
        }
        sentences.sort()
        for sentence in sentences:
            sentence_id_index = sentence_ids.index(sentence.sentence_id)

            from_location = sentence.from_location
            if from_location is None:
                from_location = result[sentence.sentence_id].before_location
            if from_location is None:
                from_location = "unk"

            to_location = sentence.to_location
            if to_location is None:
                to_location = "unk"

            if sentence.action == "CREATE":
                # propagate the lack of location upwards
                old_location = result[sentence.sentence_id].before_location
                result[sentence.sentence_id].before_location = None
                for sentence_id in reversed(sentence_ids[:sentence_id_index]):
                    if result[sentence_id].after_location != old_location:
                        break
                    result[sentence_id].after_location = None

                    if sentenceid_to_sentence.get(sentence_id, sentence_with_defaults(sentence_id)).action == 'MOVE':
                        break   # don't propagate past MOVE

                    if result[sentence_id].before_location != old_location:
                        break
                    result[sentence_id].before_location = None

                # propagate the to location downwards
                result[sentence.sentence_id].after_location = to_location
                for sentence_id in sentence_ids[sentence_id_index + 1:]:
                    result[sentence_id].before_location = to_location
                    result[sentence_id].after_location = to_location

            elif sentence.action == "DESTROY":
                # propagate the from location upwards
                old_location = result[sentence.sentence_id].before_location
                result[sentence.sentence_id].before_location = from_location
                for sentence_id in reversed(sentence_ids[:sentence_id_index]):
                    if result[sentence_id].after_location != old_location:
                        break
                    result[sentence_id].after_location = from_location

                    if sentenceid_to_sentence.get(sentence_id, sentence_with_defaults(sentence_id)).action == 'MOVE':
                        break   # don't propagate past MOVE

                    if result[sentence_id].before_location != old_location:
                        break
                    result[sentence_id].before_location = from_location

                # propagate the lack of location downwards from here
                result[sentence.sentence_id].after_location = None
                for sentence_id in sentence_ids[sentence_id_index + 1:]:
                    result[sentence_id].before_location = None
                    result[sentence_id].after_location = None

            elif sentence.action == "MOVE":
                # propagate the from location upwards
                old_location = result[sentence.sentence_id].before_location
                result[sentence.sentence_id].before_location = from_location
                for sentence_id in reversed(sentence_ids[:sentence_id_index]):
                    if result[sentence_id].after_location != old_location:
                        break
                    result[sentence_id].after_location = from_location

                    if sentenceid_to_sentence.get(sentence_id, sentence_with_defaults(sentence_id)).action == 'MOVE':
                        break   # don't propagate past MOVE

                    if result[sentence_id].before_location != old_location:
                        break
                    result[sentence_id].before_location = from_location

                # propagate the to location downwards
                result[sentence.sentence_id].after_location = to_location
                for sentence_id in sentence_ids[sentence_id_index + 1:]:
                    result[sentence_id].before_location = to_location
                    result[sentence_id].after_location = to_location

            else:
                if from_location == "unk":
                    from_location = to_location
                if to_location == "unk":
                    to_location = from_location

                # propagate the from location upwards
                old_location = result[sentence.sentence_id].before_location
                result[sentence.sentence_id].before_location = from_location
                for sentence_id in reversed(sentence_ids[:sentence_id_index]):
                    if result[sentence_id].after_location != old_location:
                        break
                    result[sentence_id].after_location = from_location

                    if sentenceid_to_sentence.get(sentence_id, sentence_with_defaults(sentence_id)).action == 'MOVE':
                        break   # don't propagate past MOVE

                    if result[sentence_id].before_location != old_location:
                        break
                    result[sentence_id].before_location = from_location

                # propagate the to location downwards
                result[sentence.sentence_id].after_location = to_location
                for sentence_id in sentence_ids[sentence_id_index + 1:]:
                    result[sentence_id].before_location = to_location
                    result[sentence_id].after_location = to_location

        yield from result.values()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply inertia to the output of this model")

    parser.add_argument(
        '--predictions', '-p',
        type=str,
        help='Filename of the model results. JSON format expected.',
        required=True)
    parser.add_argument(
        '--full-grid', '-g',
        type=str,
        help='Filename of the full grid TSV-like format expected.',
        required=True)
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file. Default is stdout.',
        type=str,
        default=None)

    args = parser.parse_args()

    with file_or_stdout(args.output) as output:
        for result in apply_inertia(args.predictions, args.full_grid):
            output.write(result.render())
            output.write("\n")
