"""Launch the local TTS API on an isolated port and validate its contract."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda','dml'],default='cuda')
    p.add_argument('--avatar',choices=['ami','tyc'],default='tyc');a=p.parse_args()
    a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    assert not (a.output/'result.json').exists()
    import numpy as np
    import soundfile as sf
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    token=secrets.token_hex(24)
    env=dict(os.environ,TEMP=str(a.output),TMP=str(a.output),PYTHONIOENCODING='utf-8')
    command=[sys.executable,str(ROOT/'api_server.py'),'--host','127.0.0.1','--port',str(port),
        '--token',token,'--avatar',a.avatar,'--reference','default','--device',a.device,'--cpu_threads','4',
        '--resources',str(ROOT/'resources')]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,payload=None):
        data=json.dumps(payload,ensure_ascii=False).encode('utf-8') if payload is not None else None
        req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with opener.open(req,timeout=180) as response:return response.read()
    def obj(path,payload=None):return json.loads(request(path,payload))
    rows=[]
    with (a.output/'server.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=str(a.output),env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+120
            while True:
                if time.monotonic()>deadline:raise TimeoutError('API startup deadline exceeded')
                if process.poll() is not None:raise RuntimeError('API exited before startup; inspect server.log')
                try:
                    if obj('/api/tts/status')['ready']:break
                except OSError:
                    if time.monotonic()>deadline:raise
                time.sleep(.5)
            for voice in ('tyc','ami'):
                for key,text in [('zh','早饭想吃什么？我打算煮一碗面，再加一个鸡蛋。'),
                                 ('mixed','今天也辛苦了，お疲れさま，have a good rest，我们明天见。')]:
                    # Same request shape as the public client: no explicit textLang.
                    result=obj('/api/tts/predict-timeline',dict(text=text,modelId=voice,referenceId='default'))
                    assert result['active_model_id']==voice and result['sample_rate']==32000 and result['timestamps']
                    wav=base64.b64decode(result.pop('audio'));x,sr=sf.read(io.BytesIO(wav),dtype='float32')
                    assert sr==32000 and x.ndim==1 and len(x)>0 and np.isfinite(x).all()
                    for t in result['timestamps']:
                        assert t['char']==text[t['char_idx']] and 0<=t['start']<=t['end']<=result['duration']+1e-5
                    (a.output/f'{voice}-{key}.wav').write_bytes(wav)
                    rows.append(dict(voice=voice,case=key,**result))
            wav=request('/api/tts/predict',dict(text='明天见。',modelId='ami'))
            assert wav[:4]==b'RIFF'
            stream_cases=[]
            for lang,text in [('zh','早饭想吃什么？我打算煮一碗面，再加一个鸡蛋。'),('ja','こんにちは、今日もよろしくお願いします。'),('en','Good morning! Would you like some tea?'),('auto','今天一起喝 coffee，然后说一句ありがとう。')]:
                stream=request('/api/tts/predict-stream',dict(text=text,modelId='ami',textLang=lang,chunkLength=12))
                events=[json.loads(line) for line in stream.splitlines() if line]
                assert events[0]['type']=='start' and events[-1]['type']=='done'
                cursor=0;timeline=[];pcm=[]
                for e in events:
                    assert e['type']!='error',e
                    if e['type']!='chunk':continue
                    x=np.frombuffer(base64.b64decode(e['audio']),dtype='<f4')
                    assert e['audio_start_sample']==cursor and np.isfinite(x).all()
                    cursor+=len(x);assert e['audio_end_sample']==cursor
                    timeline+=e['timestamps'];pcm.append(x)
                assert timeline and cursor==events[-1]['audio_end_sample']
                assert all(0<=r['start']<=r['end']<=cursor/32000.+1e-5 for r in timeline)
                sf.write(a.output/f'stream-{lang}.wav',np.concatenate(pcm),32000)
                stream_cases.append(dict(language=lang,chunks=len(pcm),timestamps=len(timeline),pre_eos=any(not e.get('gpt_finished',True) for e in events)))
            (a.output/'streams.json').write_text(json.dumps(stream_cases,indent=2),'utf-8')
            # Disconnect after one packet, then immediately request another
            # operation: the cancelled generator must release/restore the model.
            payload=json.dumps(dict(text='今天我想和你聊一聊，我们可以先去散步，然后再一起做晚饭。',modelId='ami',chunkLength=12)).encode('utf-8')
            req=urllib.request.Request(f'http://127.0.0.1:{port}/api/tts/predict-stream',data=payload,
                headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            with opener.open(req,timeout=60) as response:
                while True:
                    event=json.loads(response.readline())
                    if event['type']=='chunk':break
                    assert event['type']!='error',event
            asleep=obj('/api/tts/hibernate',{})
            assert asleep['runtime_state']=='hibernated' and not asleep['ready']
            awake=obj('/api/tts/wake',{})
            assert awake['ready'] and awake['active_model_id']=='ami'
            result=dict(passed=True,samples=rows,predict_wav=True,stream_events=[e['type'] for e in events],
                hibernate_wake=True,port=port)
            (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
            print(json.dumps(dict(passed=True,samples=len(rows),stream=result['stream_events'],hibernate_wake=True)),flush=True)
        finally:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
