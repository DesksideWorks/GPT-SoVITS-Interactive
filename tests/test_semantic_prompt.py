"""Language selection, asset integrity and prefix/target contract tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from semantic_prompt import SemanticPrompt, file_sha256, load_for_checkpoint, prepare_inputs
from ort_gpt import OrtGpt
from run_inference import GPTSoVITSInference
from streaming_latents import LatentStream


def write_prompt(root, name, language, checkpoint, phones=2, sovits=None):
    asset = root/(name+'.npz')
    np.savez(asset, phones=np.arange(1, phones+1, dtype=np.int64), semantic=np.array([4, 5], dtype=np.int64), bert=np.zeros((1024, phones), dtype=np.float32))
    manifest = dict(schema=1, language=language, semantic_frame_rate=25, id=name, file=asset.name, sha256=file_sha256(asset), gpt_sha256=file_sha256(checkpoint), phone_count=phones, token_count=2)
    if sovits: manifest['sovits_sha256'] = file_sha256(sovits)
    (root/(name+'.json')).write_text(json.dumps(manifest),'utf-8')
    return asset


class SemanticPromptTests(unittest.TestCase):
    def test_asset_integrity_and_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); checkpoint = root/'gpt'; checkpoint.write_bytes(b'accepted-gpt')
            asset = write_prompt(root,'zh-test','zh',checkpoint)
            prompts = load_for_checkpoint(checkpoint,root)
            self.assertEqual(['zh'],list(prompts));prompt = prompts['zh']
            self.assertEqual('zh-test',prompt.identifier)
            self.assertEqual(2,prompt.token_count);self.assertEqual('cpu',prompt.bert.device.type)
            other=root/'other';other.write_bytes(b'other-gpt')
            self.assertEqual({},load_for_checkpoint(other,root))
            asset.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):load_for_checkpoint(checkpoint,root)

    def test_one_prompt_per_language(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); checkpoint = root/'gpt'; checkpoint.write_bytes(b'accepted-gpt')
            other=root/'other';other.write_bytes(b'other-gpt')
            write_prompt(root,'zh-test','zh',checkpoint);write_prompt(root,'en-test','en',checkpoint,phones=3)
            write_prompt(root,'ja-test','ja',checkpoint);write_prompt(root,'zh-other-gpt','zh',other)
            prompts = load_for_checkpoint(checkpoint,root)
            self.assertEqual({'zh':'zh-test','en':'en-test','ja':'ja-test'},{k:v.identifier for k,v in prompts.items()})
            self.assertEqual(3,prompts['en'].phone_count)
            write_prompt(root,'zh-second','zh',checkpoint)
            with self.assertRaisesRegex(ValueError,'More than one zh'):load_for_checkpoint(checkpoint,root)
            (root/'bad.json').write_text(json.dumps(dict(schema=1,language='ko',semantic_frame_rate=25,file='bad.npz')),'utf-8')
            with self.assertRaisesRegex(ValueError,'manifest'):load_for_checkpoint(other,root)

    def test_character_bound_prompt_serves_one_character(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); checkpoint = root/'gpt'; checkpoint.write_bytes(b'accepted-gpt')
            ami=root/'ami';ami.write_bytes(b'ami-sovits');tyc=root/'tyc';tyc.write_bytes(b'tyc-sovits')
            write_prompt(root,'zh-shared','zh',checkpoint)
            write_prompt(root,'ja-ami','ja',checkpoint,sovits=ami);write_prompt(root,'ja-tyc','ja',checkpoint,sovits=tyc)
            ids=lambda sovits:{k:v.identifier for k,v in load_for_checkpoint(checkpoint,root,sovits).items()}
            self.assertEqual({'zh':'zh-shared','ja':'ja-ami'},ids(ami))
            self.assertEqual({'zh':'zh-shared','ja':'ja-tyc'},ids(tyc))
            self.assertEqual({'zh':'zh-shared'},ids(None))
            write_prompt(root,'ja-shared','ja',checkpoint)
            self.assertEqual({'zh':'zh-shared','ja':'ja-shared'},ids(None))
            with self.assertRaisesRegex(ValueError,'More than one ja'):ids(ami)

    def test_prompt_follows_frontend_language(self):
        engine=GPTSoVITSInference.__new__(GPTSoVITSInference);prompts=dict(zh=object(),ja=object(),en=object())
        cases=[('今天晚上吃什么？','zh','zh'),('今天晚上吃什么？','auto','zh'),('今天晚上吃什么？','all_zh','zh'),
               ('今日は何を食べますか。','ja','ja'),('今日は何を食べますか。','auto','ja'),('東京都','ja','ja'),('東京都','auto','zh'),
               ('Have some tea.','en','en'),('Have some tea.','auto','en'),('Have some tea.','ja','en'),('今天吃什么？','en',None),
               ('今天喝 coffee。','zh',None),('今天休息。明天说 hello。','auto',None),('你好、ありがとう。','auto',None),
               ('OKだよ、まかせて。','ja',None),('……','zh',None),('……','auto',None),('','auto',None),('123','zh',None),
               # A Japanese name inside Chinese keeps the Chinese prefix; without it Chinese can stall in silence.
               ('对了，我叫コハルネアミ。两个人在一块儿，日子还挺热闹。','auto','zh'),('我叫アミ。','zh','zh'),
               ('和ツクヨミちゃん一起去。','auto','zh'),('今天也辛苦了，お疲れさま。','auto','zh'),
               ('はじめまして、コハルネアミです。','auto','ja')]
        for text,language,expected in cases:
            for available in (prompts,dict(zh=prompts['zh'])):
                engine._semantic_prompts=available
                with self.subTest(text=text,language=language,available=list(available)):
                    self.assertIs(available.get(expected),engine._semantic_prompt_for_text(text,language))
        engine._semantic_prompts={}
        self.assertIsNone(engine._semantic_prompt_for_text('今天晚上吃什么？','zh'))
        engine._semantic_prompts=prompts
        with self.assertRaisesRegex(ValueError,'Unsupported TTS language'):engine._semantic_prompt_for_text('今天','ko')

    def test_prefixed_request_keeps_one_frontend_language(self):
        class Stop(Exception):pass
        seen=[]
        def segments(text,language):seen.append(language);raise Stop
        engine=GPTSoVITSInference.__new__(GPTSoVITSInference);engine.gpt_backend=engine.backend=mock.Mock()
        engine._semantic_prompts=dict(ja=SemanticPrompt('ja-test',torch.tensor([1]),torch.tensor([2]),torch.zeros(1024,1),'ja'))
        engine._text_segments=segments
        stream=LatentStream.__new__(LatentStream);stream.engine=engine;stream.condition=lambda:(None,None)
        # A kanji-only sentence detected on its own would be read as Chinese.
        for text,expected in [('今日はいい天気ですね。東京都庁。','ja'),('今天喝 coffee。','auto')]:
            with self.subTest(text=text):
                with self.assertRaises(Stop):engine.synthesize(text,'auto')
                with self.assertRaises(Stop):list(stream._stream(text,'auto'))
                self.assertEqual([expected]*2,seen);seen.clear()

    def test_inputs_do_not_modify_target_or_cpu_prompt(self):
        prompt=SemanticPrompt('test',torch.tensor([1,2]),torch.tensor([3,4,5]),torch.ones(1024,2))
        meta=dict(phones=[8,9,10],bert=torch.zeros(1024,3))
        x,l,p,b=prepare_inputs(meta,'cpu',torch.float16,prompt)
        self.assertEqual([[1,2,8,9,10]],x.tolist());self.assertEqual([5],l.tolist())
        self.assertEqual([[3,4,5]],p.tolist());self.assertEqual((1,1024,5),tuple(b.shape))
        self.assertEqual([8,9,10],meta['phones']);self.assertEqual(torch.float32,prompt.bert.dtype)

    def test_ort_full_includes_prefix_stream_does_not(self):
        g=OrtGpt.__new__(OrtGpt);torch.nn.Module.__init__(g)
        g._tokens=lambda *args,**kwargs:iter(range(51))
        prefix=torch.tensor([[901,902]])
        full,_=g.infer_panel(None,None,prefix,None)
        chunks=list(g.infer_panel_naive(None,None,prefix,None,streaming_mode=True,chunk_length=12))
        target=torch.cat([t for t,_ in chunks if t is not None],dim=1)
        self.assertEqual([[901,902]],full[:,:2].tolist());self.assertTrue(torch.equal(full[:,2:],target))
        self.assertTrue(chunks[-1][1])

    def test_prefixed_budget_preserves_generated_capacity(self):
        g=OrtGpt.__new__(OrtGpt);torch.nn.Module.__init__(g);g.capacity=512;g.prefill_size=256
        self.assertEqual(180,g.segment_phone_budget())
        budget=g.segment_phone_budget(42,97)
        self.assertEqual(85,budget);self.assertLessEqual(42+97+4*budget+32,512)
        with self.assertRaises(ValueError):g.segment_phone_budget(256,256)


if __name__=='__main__':unittest.main()
