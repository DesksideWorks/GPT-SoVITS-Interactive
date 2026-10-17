"""Opt-in real-weight DirectML parity, timeline, voice-switch and lifecycle check."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import soundfile as sf
from run_inference import GPTSoVITSInference
from GPT_SoVITS.module import models


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    voices=json.loads((ROOT/'resources/voices.json').read_text('utf-8'))
    prompts=[('zh','早饭想吃什么？我打算煮一碗面，再加一个鸡蛋。'),
             ('ja','こんにちは、今日もよろしくお願いします。'),
             ('en','Good morning! Would you like some tea?'),
             ('auto','今天一起喝 coffee，然后说一句ありがとう。')]
    original=models.randn_like_backend
    models.randn_like_backend=lambda t: torch.randn(t.shape,generator=torch.Generator().manual_seed(9000),dtype=t.dtype).to(t.device)
    reports=[]
    try:
        for voice,c in voices.items():
            expected={}
            for route in ('cpu','dml'):
                engine=GPTSoVITSInference(str(ROOT/'resources'/c['gpt_path']),str(ROOT/'resources'/c['sovits_path']),
                    str(ROOT/'resources'/c['references']['default']['cache_path']),device='dml',gpt_device=route,cpu_threads=4,pipeline=False)
                try:
                    traces=[];panel=engine.t2s_model.model.infer_panel
                    def trace(*a,**k):
                        value=panel(*a,**k);traces.append(value[0].cpu().tolist());return value
                    engine.t2s_model.model.infer_panel=trace
                    for i,(lang,text) in enumerate(prompts):
                        for r in range(2):
                            torch.manual_seed(42);traces.clear();started=time.perf_counter()
                            wave,timeline=engine.synthesize(text,**dict(c['defaults'],text_lang=lang,return_timeline=True,pipeline=False))
                            assert np.isfinite(wave).all() and len(wave)>0 and timeline
                            result=dict(voice=voice,route=route,prompt=i,round=r,seconds=time.perf_counter()-started,metrics=engine.last_metrics)
                            if r==1:
                                folder=out/f'{voice}-{route}-{i}';folder.mkdir(exist_ok=True)
                                sf.write(folder/'audio.wav',wave,engine.sample_rate)
                                (folder/'timeline.json').write_text(json.dumps(timeline,ensure_ascii=False),'utf-8')
                                if route=='cpu':expected[i]=(wave.copy(),timeline,list(traces))
                                else:
                                    ref,ref_timeline,ref_tokens=expected[i]
                                    assert traces==ref_tokens, 'semantic tokens differ'
                                    np.testing.assert_array_equal(wave,ref)
                                    assert timeline==ref_timeline
                                    result['parity']=True
                            reports.append(result)
                    if route=='dml':
                        for _ in range(2):
                            engine.hibernate();engine.hibernate();assert engine.t2s_model.pre is None
                            engine.wake();engine.wake();assert engine.t2s_model.pre is not None
                            torch.manual_seed(42);traces.clear()
                            wave,timeline=engine.synthesize(prompts[0][1],**dict(c['defaults'],text_lang='zh',return_timeline=True,pipeline=False))
                            np.testing.assert_array_equal(wave,expected[0][0]);assert timeline==expected[0][1]
                        reports.append(dict(voice=voice,lifecycle_cycles=2,adapter=engine.t2s_model.adapter))
                finally:engine.close()
        (out/'result.json').write_text(json.dumps(dict(passed=True,rows=reports),ensure_ascii=False,indent=2),'utf-8')
    finally:models.randn_like_backend=original


if __name__=='__main__':main()
