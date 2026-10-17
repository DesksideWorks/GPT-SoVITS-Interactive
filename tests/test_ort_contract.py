"""Backend routing and graph-contract regression without model weights."""
import sys
from pathlib import Path
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dxgi_adapters import match_adapter
from ort_gpt import OrtGpt
from run_inference import GPTSoVITSInference


class OrtContractTests(unittest.TestCase):
    def test_physical_adapter_order_is_not_torch_order(self):
        adapters = [dict(index=0,name='AMD Radeon RX 7800 XT',software=False,luid='a'),
                    dict(index=1,name='AMD Radeon(TM) Graphics',software=False,luid='b')]
        self.assertEqual(0,match_adapter('AMD Radeon RX 7800 XT\x00',adapters)['index'])
        self.assertEqual(1,match_adapter('AMD Radeon(TM) Graphics',adapters)['index'])

    def test_ambiguous_adapter_requires_identity(self):
        adapters = [dict(index=i,name='GPU',software=False,luid=str(i)) for i in range(2)]
        with self.assertRaises(RuntimeError):match_adapter('GPU',adapters)
        with mock.patch('gpu_adapter_probe.torch_adapter_luid',return_value='1'):
            self.assertEqual(1,match_adapter('GPU',adapters,torch_device='privateuseone:0')['index'])

    def test_hibernate_drops_both_sessions(self):
        g = OrtGpt.__new__(OrtGpt);torch.nn.Module.__init__(g)
        g.pre=object();g.step=object();g.hibernate();g.hibernate()
        self.assertIsNone(g.pre);self.assertIsNone(g.step)

    def test_segment_fitting_preserves_all_text(self):
        engine=GPTSoVITSInference.__new__(GPTSoVITSInference)
        engine.t2s_model=mock.Mock(max_segment_phones=8)
        engine.hps=mock.Mock();engine.hps.model.version='v2ProPlus'
        engine._prompt_cache=mock.Mock();engine._prompt_cache.metadata.ref_lang='ja'
        engine.get_phones_and_bert_with_metadata=lambda text,*a,**k: {'phones':list(text)}
        text='今天我们一起去超市买些蔬菜，再回家准备晚饭。'
        parts=engine._text_segments(text,'zh')
        self.assertEqual(text,''.join(parts));self.assertTrue(all(len(p)<=8 for p in parts))

    def test_stream_and_complete_share_token_generation(self):
        g = OrtGpt.__new__(OrtGpt);torch.nn.Module.__init__(g)
        g._tokens=lambda *a,**k: iter(range(51))
        complete,_=g.infer_panel(None,None,None,None)
        chunks=list(g.infer_panel_naive(None,None,None,None,streaming_mode=True,chunk_length=24))
        self.assertTrue(chunks[-1][1]);self.assertTrue(torch.equal(complete,torch.cat([c for c,_ in chunks if c is not None],1)))


if __name__=='__main__':unittest.main()
