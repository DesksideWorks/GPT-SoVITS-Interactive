import pathlib
import sys
import unittest

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from timeline import (  # noqa: E402
    StreamingTimeline,
    finalize_timestamps,
    monotonic_phone_path,
    process_attention,
    scale_timestamps_to_duration,
)


def timestamp(char_idx, char, start, end):
    return {
        "char_idx": char_idx,
        "ph_start": char_idx,
        "ph_end": char_idx + 1,
        "phones": [char],
        "start": start,
        "end": end,
        "char": char,
    }


def voice_attention(spans, phone_count):
    """Attention as the voice model gives it: a list of (phone, frames), ``None`` for silence.

    The last phone keeps part of the attention during speech and takes nearly all of it in silence.
    """

    rows = []
    for phone, frames in spans:
        row = np.full((phone_count,), 0.1 / phone_count)
        if phone is None:
            row[-1] += 0.9
        else:
            row[phone] += 0.6
            row[-1] += 0.3
        rows.extend([row] * frames)
    return np.asarray(rows)


class TimelineTests(unittest.TestCase):
    def test_monotonic_path_never_moves_backwards(self):
        attention = np.asarray(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.8, 0.1, 0.1],  # independent argmax would jump back
                [0.0, 0.2, 0.8],
            ],
            dtype=np.float32,
        )
        path = monotonic_phone_path(attention, force_complete=True)
        self.assertTrue(np.all(path[1:] >= path[:-1]))
        self.assertEqual(0, int(path[0]))
        self.assertEqual(2, int(path[-1]))

    def test_single_phone_attention_does_not_crash(self):
        path = monotonic_phone_path(np.ones((4, 1), dtype=np.float32))
        np.testing.assert_array_equal(path, np.zeros(4, dtype=np.int64))

    def test_repeated_characters_keep_distinct_source_indices(self):
        raw = [
            {"char_idx": 0, "ph_start": 0, "ph_end": 1, "phones": ["k"], "start": 0.0, "end": 0.1},
            {"char_idx": 1, "ph_start": 1, "ph_end": 2, "phones": ["k"], "start": 0.1, "end": 0.2},
        ]
        result = finalize_timestamps(raw, "ここ", original_text="ここ")
        self.assertEqual([0, 1], [item["char_idx"] for item in result])
        self.assertEqual(["こ", "こ"], [item["char"] for item in result])

    def test_segment_character_offset_preserves_global_source_index(self):
        raw = [
            {"char_idx": 0, "ph_start": 0, "ph_end": 1, "phones": ["k"], "start": 0.0, "end": 0.1},
        ]
        result = finalize_timestamps(raw, "こ", original_text="こ", char_offset=3)
        self.assertEqual(3, result[0]["char_idx"])

    def test_character_dropped_by_normalization_gets_an_empty_entry(self):
        # "……" is read as a single "…"; the other one still has its index.
        raw = [
            {"char_idx": 0, "ph_start": 0, "ph_end": 2, "phones": ["AA", "ai4"], "start": 0.1, "end": 0.5},
            {"char_idx": 1, "ph_start": 2, "ph_end": 3, "phones": ["…"], "start": 0.5, "end": 0.9},
            {"char_idx": 2, "ph_start": 3, "ph_end": 5, "phones": ["h", "ao3"], "start": 0.9, "end": 1.2},
        ]
        result = finalize_timestamps(raw, "唉…好", original_text="唉……好", char_offset=4)
        self.assertEqual([4, 5, 6, 7], [item["char_idx"] for item in result])
        self.assertEqual("唉……好", "".join(item["char"] for item in result))
        empty = [item for item in result if not item["phones"]]
        self.assertEqual(1, len(empty))
        self.assertEqual(empty[0]["start"], empty[0]["end"])
        for before, after in zip(result, result[1:]):
            self.assertLessEqual(before["end"], after["start"])

    def test_dropped_character_waits_for_the_character_after_it(self):
        # An unfinished alignment must not hand out what follows its last character.
        spoken = {"char_idx": 0, "ph_start": 0, "ph_end": 1, "phones": ["a"], "start": 0.0, "end": 0.2}
        later = {"char_idx": 1, "ph_start": 1, "ph_end": 2, "phones": ["b"], "start": 0.3, "end": 0.5}
        self.assertEqual("a", "".join(item["char"] for item in finalize_timestamps([spoken], "ab", original_text="a*b*")))
        result = finalize_timestamps([spoken, later], "ab", original_text="a*b*")
        self.assertEqual("a*b*", "".join(item["char"] for item in result))
        self.assertEqual([0.2, 0.5], [item["start"] for item in result if item["char"] == "*"])

    def test_dropped_character_before_the_first_one_starts_with_it(self):
        spoken = {"char_idx": 0, "ph_start": 0, "ph_end": 1, "phones": ["a"], "start": 0.3, "end": 0.5}
        result = finalize_timestamps([spoken], "a", original_text="*a")
        self.assertEqual([0.3, 0.3], [item["start"] for item in result])
        self.assertEqual([0.3, 0.5], [item["end"] for item in result])

    def test_full_timeline_scales_to_emitted_audio_duration(self):
        source = [timestamp(0, "こ", 0.5, 0.7), timestamp(1, "こ", 0.7, 1.0)]
        result = scale_timestamps_to_duration(source, duration_seconds=1.5, offset_seconds=0.5)
        self.assertAlmostEqual(0.5, result[0]["start"])
        self.assertAlmostEqual(2.0, result[-1]["end"])

    def test_process_attention_emits_monotonic_character_ranges(self):
        attention = np.asarray(
            [[[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]]],
            dtype=np.float32,
        )
        result = process_attention(attention, [[1, 1]], [4], 1.0, [["a", "b"]])[0]
        self.assertLessEqual(result[0]["start"], result[0]["end"])
        self.assertLessEqual(result[0]["end"], result[1]["end"])

    def test_incremental_attention_omits_unspoken_trailing_characters(self):
        attention = np.asarray(
            [[[[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]]]],
            dtype=np.float32,
        )
        result = process_attention(
            attention,
            [[1, 1, 1]],
            [2],
            1.0,
            [["a", "b", "c"]],
            force_complete=False,
        )[0]
        self.assertEqual(2, len(result))

    def test_incremental_attention_of_silence_alone_names_no_character(self):
        result = process_attention(
            voice_attention([(None, 12)], 4),
            [[1, 1, 1, 1]],
            [12],
            1.0,
            [["a", "b", "c", "."]],
            force_complete=False,
        )[0]
        self.assertEqual([], result)

    def test_silence_before_the_first_word_belongs_to_no_character(self):
        # Silent frames attend to the end of the text.  Following them there
        # used to press every character into the silence, one frame each.
        attention = voice_attention([(None, 20), (0, 6), (1, 6), (2, 6), (3, 4)], 4)
        result = process_attention(attention, [[1, 1, 1, 1]], [42], 1.0, [["a", "b", "c", "."]])[0]
        self.assertEqual([0.4, 0.52, 0.64, 0.76], [round(item["start"], 2) for item in result])
        self.assertEqual([0.52, 0.64, 0.76, 0.84], [round(item["end"], 2) for item in result])

    def test_text_without_punctuation_keeps_its_leading_silence_out(self):
        attention = voice_attention([(None, 8), (0, 6), (1, 6)], 2)
        result = process_attention(attention, [[1, 1]], [20], 1.0, [["a", "b"]])[0]
        self.assertEqual([0.16, 0.28], [round(item["start"], 2) for item in result])
        self.assertAlmostEqual(0.4, result[-1]["end"])

    def test_pause_belongs_to_its_punctuation_mark(self):
        # The comma gets the pause although the silence attends to the full stop.
        attention = voice_attention([(0, 6), (None, 10), (2, 6), (3, 4)], 4)
        result = process_attention(attention, [[1, 1, 1, 1]], [26], 1.0, [["a", ",", "b", "."]])[0]
        self.assertEqual([0.0, 0.12, 0.32, 0.44], [round(item["start"], 2) for item in result])
        self.assertEqual([0.12, 0.32, 0.44, 0.52], [round(item["end"], 2) for item in result])

    def test_pause_without_punctuation_is_a_gap_between_characters(self):
        attention = voice_attention([(0, 6), (None, 10), (1, 6), (2, 4)], 3)
        result = process_attention(attention, [[1, 1, 1]], [26], 1.0, [["a", "b", "."]])[0]
        self.assertAlmostEqual(0.12, result[0]["end"])
        self.assertAlmostEqual(0.32, result[1]["start"])


class StreamingTimelineTests(unittest.TestCase):
    SYMBOLS = ["a", "b", "c", "."]

    def stream(self, word2ph=(1, 1, 1, 1), norm_text="abc.", **options):
        symbols = self.SYMBOLS[: sum(word2ph) - 1] + ["."]
        return StreamingTimeline(list(word2ph), symbols, norm_text, **options)

    def test_evaluation_that_saw_only_silence_fixes_nothing(self):
        stream = self.stream()
        self.assertEqual([], stream.collect(voice_attention([(None, 12)], 4), 4))
        complete = voice_attention([(None, 12), (0, 6), (1, 6), (2, 6), (3, 4)], 4)
        result = stream.collect(complete, 34, final=True)
        self.assertEqual("abc.", "".join(item["char"] for item in result))
        self.assertEqual([0.24, 0.36, 0.48, 0.6], [round(item["start"], 2) for item in result])

    def test_character_waits_for_the_audio_that_holds_its_start(self):
        stream = self.stream(offset_seconds=1.0, original_text="abc.", char_offset=5)
        attention = voice_attention([(0, 6), (1, 6), (2, 6), (3, 6)], 4)
        early = stream.collect(attention, 7)
        self.assertEqual([5, 6], [item["char_idx"] for item in early])
        self.assertAlmostEqual(1.12, early[1]["start"])
        late = stream.collect(attention, 24, final=True)
        self.assertEqual([7, 8], [item["char_idx"] for item in late])

    def test_character_at_the_edge_of_the_evaluation_is_not_handed_out(self):
        # "b" is still being spoken where the evaluation ends; its end is not known.
        stream = self.stream()
        result = stream.collect(voice_attention([(0, 6), (1, 6)], 4), 12)
        self.assertEqual("a", "".join(item["char"] for item in result))

    def test_every_character_is_handed_out_once_and_in_order(self):
        stream = self.stream()
        first = stream.collect(voice_attention([(None, 2), (0, 6), (1, 6), (2, 6)], 4), 10)
        # The newer evaluation places everything earlier than what is already out.
        second = stream.collect(voice_attention([(0, 4), (1, 4), (2, 6), (3, 6)], 4), 20, final=True)
        result = first + second
        self.assertEqual([0, 1, 2, 3], [item["char_idx"] for item in result])
        for before, after in zip(result, result[1:]):
            self.assertLessEqual(before["end"], after["start"])
            self.assertLessEqual(after["start"], after["end"])

    def test_character_without_phones_is_handed_out_too(self):
        # A space has no phone.  It still has a place in the text.
        stream = self.stream(word2ph=(1, 0, 1, 1), norm_text="a b.")
        attention = voice_attention([(0, 6), (1, 6), (2, 6)], 3)
        result = stream.collect(attention, 9) + stream.collect(attention, 18, final=True)
        self.assertEqual("a b.", "".join(item["char"] for item in result))
        self.assertEqual(result[1]["start"], result[1]["end"])
        self.assertAlmostEqual(result[0]["end"], result[1]["start"])

    def test_speed_change_keeps_time_in_seconds_of_audio(self):
        # At double speed two attention rows make one 20 ms audio frame.
        stream = self.stream(speed_factor=2.0)
        result = stream.collect(voice_attention([(0, 8), (1, 8), (2, 8), (3, 8)], 4), 16, final=True)
        self.assertEqual([0.0, 0.08, 0.16, 0.24], [round(item["start"], 2) for item in result])
        self.assertAlmostEqual(0.32, result[-1]["end"])


if __name__ == "__main__":
    unittest.main()
