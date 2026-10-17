import sys
from pathlib import Path
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_inference import GPTSoVITSInference,LangSegmenter,language_spans,split_text


class MultilingualTests(unittest.TestCase):
    def test_chinese_numbers_keep_their_sentence_language(self):
        text='现在是下午3点25分，我们还有半个小时，可以慢慢走。'
        parts=LangSegmenter.getTexts(text)
        self.assertEqual([{'lang':'zh','text':text}],parts)

    def test_japanese_request_resolves_kanji_ambiguity(self):
        self.assertEqual([{'lang':'ja','text':'東京'}],LangSegmenter.getTexts('東京','ja'))

    def test_three_languages_are_preserved_without_voice_reference_hint(self):
        text='今天也辛苦了，お疲れさま，have a good rest，我们明天见。'
        parts=LangSegmenter.getTexts(text)
        self.assertEqual(['zh','ja','en','zh'],[r['lang'] for r in parts])
        self.assertEqual(text,''.join(r['text'] for r in parts))

    def test_japanese_words_in_chinese_are_read_in_japanese_and_nothing_else(self):
        cases=[('对了，我叫コハルネアミ。两个人在一块儿，日子还挺热闹。','auto',
                [('zh','对了，我叫'),('ja','コハルネアミ。'),('zh','两个人在一块儿，日子还挺热闹。')]),
               ('和ツクヨミちゃん一起去。','auto',[('zh','和'),('ja','ツクヨミちゃん'),('zh','一起去。')]),
               ('今天也辛苦了，お疲れさま。','auto',[('zh','今天也辛苦了，'),('ja','お疲れさま。')]),
               ('我是アミ，3点见。','auto',[('zh','我是'),('ja','アミ，'),('zh','3点见。')]),
               ('或者和阿米ちゃん在客厅待着。','auto',[('zh','或者和阿米'),('ja','ちゃん'),('zh','在客厅待着。')]),
               # Asked for in Chinese, kana still have only a Japanese reading.
               ('我叫アミ。','zh',[('zh','我叫'),('ja','アミ。')])]
        for text,language,expected in cases:
            with self.subTest(text=text,language=language):
                self.assertEqual(expected,language_spans(text,language))

    def test_japanese_sentences_stay_japanese(self):
        for text in ('はじめまして、コハルネアミです。駅前のカフェで働いてます。','大丈夫、心配しないで。',
                     'アミちゃんと一緒にいると楽しい。','今日はいい天気ですね。東京都庁。'):
            with self.subTest(text=text):
                self.assertEqual([('ja',text)],language_spans(text,'auto'))
        self.assertEqual([('ja','我叫アミ。')],language_spans('我叫アミ。','ja'))

    def test_merged_sentences_keep_the_original_text_between_them(self):
        # Streaming finds every piece in the request text; "Hi!How are you?" used to fail there.
        cases=[('Hi! How are you?',['Hi! How are you?']),
               ('你好！ 今天怎么样？',['你好！ 今天怎么样？']),
               ('  おはよう。\nいい天気だね。\n',['おはよう。\nいい天気だね。']),
               ('Thanks! See you tomorrow, okay?',['Thanks!','See you tomorrow, okay?'])]
        for text,expected in cases:
            pieces=split_text(text)
            self.assertEqual(expected,pieces)
            offset=0
            for piece in pieces:
                offset=text.index(piece,offset)+len(piece)

    def test_ref_free_sentinel_does_not_discard_tokens(self):
        engine=GPTSoVITSInference.__new__(GPTSoVITSInference)
        engine.gpt_device='cpu';engine.gpt_is_half=False;engine.gpt_backend=mock.Mock()
        engine.hps=types.SimpleNamespace(model=types.SimpleNamespace(version='v2ProPlus'))
        meta=dict(phones=[11,12],bert=torch.zeros(1024,2),norm_text='你好')
        engine.get_phones_and_bert_with_metadata=mock.Mock(return_value=meta)
        panel=mock.Mock(return_value=(torch.tensor([[7,8,9]]),0))
        engine.t2s_model=types.SimpleNamespace(model=types.SimpleNamespace(infer_panel=panel))
        result=engine._generate_segment_semantic('你好','auto','ja',15,1,1)
        args=panel.call_args.args
        self.assertIsNone(args[2])
        self.assertEqual([[11,12]],args[0].tolist())
        self.assertEqual([1,1024,2],list(args[3].shape))
        self.assertEqual([[[7,8,9]]],result['pred_semantic'].tolist())
        self.assertEqual(3,result['idx'])


if __name__=='__main__':unittest.main()
