import os
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
# The English frontend reads its tagger from here, as the runtime sets it up.
os.environ.setdefault("NLTK_DATA", str(ROOT / "resources" / "nltk_data"))

import timeline  # noqa: E402
from GPT_SoVITS.text.cleaner import clean_text  # noqa: E402
from timeline import english_word2ph, japanese_word2ph  # noqa: E402


# Accent marks depend on the dictionary's accent data; the tests look at phonemes.
ACCENT_MARKS = ("[", "]", "UNK")


def mapping(text, language="ja"):
    """The phonemes each character of ``text`` was given."""
    phones, _, norm_text = clean_text(text, language, "v2")
    counts = (japanese_word2ph if language == "ja" else english_word2ph)(norm_text, phones)
    result = []
    position = 0
    for char, count in zip(norm_text, counts):
        spoken = [phone for phone in phones[position : position + count] if phone not in ACCENT_MARKS]
        result.append((char, " ".join(spoken)))
        position += count
    return norm_text, phones, counts, result


class JapaneseWord2phTests(unittest.TestCase):
    def assert_covers(self, text):
        norm_text, phones, counts, _ = mapping(text)
        self.assertEqual(len(norm_text), len(counts), text)
        self.assertEqual(len(phones), sum(counts), text)
        self.assertTrue(all(count >= 0 for count in counts), text)

    def test_every_phone_belongs_to_one_character(self):
        with mock.patch.object(timeline, "print", create=True) as approximate:
            for text in (
                "こんにちは！はじめまして、だよね。",
                "好きなものはいっぱいあるよ。いれたてのコーヒー、静かな部屋、それから道ばたの猫。",
                "2026年10月13日、３時２５分。",
                "Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！",
                "えーと…あぁ、うぉー！",
                "100％だよ",
                "コハルネ・アミです",
                "「画面の外のものには、なんにも触れないんだ。」",
                "ｶﾀｶﾅ",
                "あ い",
                "ー",
                "々",
                "。",
                " ",
            ):
                self.assert_covers(text)
        approximate.assert_not_called()

    def test_kana_follow_their_moras(self):
        _, _, _, result = mapping("こんにちは！")
        self.assertEqual(
            [("こ", "k o"), ("ん", "N"), ("に", "n i"), ("ち", "ch i"), ("は", "w a"), ("！", "!")],
            result,
        )

    def test_small_kana_shares_the_mora_of_the_kana_before_it(self):
        _, _, _, result = mapping("きゃりー")
        self.assertEqual([("き", "ky"), ("ゃ", "a"), ("り", "r i"), ("ー", "i")], result)

    def test_small_kana_can_be_a_mora_of_its_own(self):
        _, _, _, result = mapping("あぁ")
        self.assertEqual([("あ", "a"), ("ぁ", "a")], result)

    def test_kanji_take_what_the_kana_of_the_word_leave(self):
        _, _, _, result = mapping("始まってる")
        self.assertEqual(
            [("始", "h a j i"), ("ま", "m a"), ("っ", "cl"), ("て", "t e"), ("る", "r u")],
            result,
        )

    def test_respelled_digits_keep_the_following_word_exact(self):
        _, _, _, result = mapping("1000円")
        self.assertEqual(("円", "e N"), result[-1])
        self.assertEqual("s e N", " ".join(phones for _, phones in result[:-1] if phones))

    def test_spelled_out_symbol_maps_back_to_one_character(self):
        _, _, _, result = mapping("100％だよ")
        self.assertEqual(("％", "p a a s e N t o"), result[3])
        self.assertEqual([("だ", "d a"), ("よ", "y o")], result[4:])

    def test_pause_stays_with_the_symbol_that_caused_it(self):
        _, _, _, result = mapping("ネ・アミ")
        self.assertEqual(("・", "_"), result[1])
        self.assertTrue(result[2][1].startswith("a"))

    def test_unexpected_phone_count_spreads_instead_of_failing(self):
        phones, _, norm_text = clean_text("こんにちは", "ja", "v2")
        with mock.patch.object(timeline, "print", create=True) as approximate:
            counts = japanese_word2ph(norm_text, phones + ["a", "a"])
        approximate.assert_called_once()
        self.assertEqual(len(norm_text), len(counts))
        self.assertEqual(len(phones) + 2, sum(counts))


class EnglishWord2phTests(unittest.TestCase):
    def words(self, text):
        """The phonemes of every run of characters between spaces."""
        norm_text, _, _, result = mapping(text, "en")
        groups = [[]]
        for char, phones in result:
            if char == " ":
                groups.append([])
            elif phones:
                groups[-1].append(phones)
        return norm_text, [" ".join(group) for group in groups]

    def test_every_phone_belongs_to_one_character(self):
        with mock.patch.object(timeline, "print", create=True) as approximate:
            for text in (
                "Hello there! Nice to meet you, right?",
                "I don't know... it's 3 o'clock, e.g. tea-time.",
                "She said \"wow\" at DesktopDaily's GPU-demo!!",
                "uh-huh — well; yes: 25% of AI",
                "Hi.",
                "OK",
                "A",
                "...",
                " ",
            ):
                norm_text, phones, counts, _ = mapping(text, "en")
                self.assertEqual(len(norm_text), len(counts), text)
                self.assertEqual(len(phones), sum(counts), text)
                self.assertTrue(all(count >= 0 for count in counts), text)
        approximate.assert_not_called()

    def test_words_keep_their_own_phones(self):
        _, words = self.words("Nice to meet you, right?")
        self.assertEqual(["N AY1 S", "T UW1", "M IY1 T", "Y UW1 ,", "R AY1 T ?"], words)

    def test_respelled_text_maps_through_its_normal_form(self):
        norm_text, words = self.words("it's 3 o'clock")
        self.assertEqual("it's three o'clock", norm_text)
        self.assertEqual(["IH1 T S", "TH R IY1", "AH0 K L AA1 K"], words)

    def test_a_word_starts_on_its_first_letter(self):
        norm_text, _, counts, _ = mapping("Hello there! Nice to meet you, right?", "en")
        for index, char in enumerate(norm_text):
            if char == " ":
                self.assertEqual(0, counts[index])
            elif index == 0 or norm_text[index - 1] == " ":
                self.assertGreater(counts[index], 0, norm_text[index:])

    def test_the_comma_added_to_a_short_text_stays_on_the_first_character(self):
        _, phones, counts, result = mapping("Hi.", "en")
        self.assertEqual([",", "HH", "AY1", "."], list(phones))
        self.assertEqual(("H", ", HH"), result[0])

    def test_unexpected_phone_count_spreads_instead_of_failing(self):
        phones, _, norm_text = clean_text("Nice to meet you.", "en", "v2")
        with mock.patch.object(timeline, "print", create=True) as approximate:
            counts = english_word2ph(norm_text, phones + ["AH0", "AH0"])
        approximate.assert_called_once()
        self.assertEqual(len(norm_text), len(counts))
        self.assertEqual(len(phones) + 2, sum(counts))


if __name__ == "__main__":
    unittest.main()
