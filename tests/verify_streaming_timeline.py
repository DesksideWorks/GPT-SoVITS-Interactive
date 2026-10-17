"""Real-weight streaming contract and timeline side-channel parity checks."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import soundfile as sf
from run_inference import GPTSoVITSInference
from streaming_latents import LatentStream
from timeline import finalize_timestamps
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
route=sys.argv[2] if len(sys.argv)>2 else 'cuda'
voices=json.loads((ROOT/'resources/voices.json').read_text('utf-8'))
reports=[]
# Lines that open with a pause or an interjection, where the characters used to be
# pressed into the silence before the voice, and one without any punctuation.
LINES=[('zh','唉……今天好累啊。'),('zh','嗯……要不要先喝点什么？'),('zh','对不起'),('ja','うん、わかった！'),('ja','えっと……どうしようかな。'),('en','Hmm... let me think about it.')]
SILENT=set(' !?…,.-，。！？、')
def check_characters(engine,voice,language,text):
 """Every character once and in order, and the characters where the voice is."""
 torch.manual_seed(42);packets=[];rows=[];bounds={}
 for event in LatentStream(engine,seed=9000).stream(text,language=language):
  if event['type']!='chunk':continue
  if not event.get('pause'):
   first,_=bounds.get(event['segment'],(event['start_sample'],0));bounds[event['segment']]=(first,event['start_sample']+event['sample_count'])
  packets.append(event['pcm']);rows+=event['timestamps']
 wave=np.concatenate(packets)
 assert all(a['char_idx']<b['char_idx'] and a['end']<=b['start']+1e-6 for a,b in zip(rows,rows[1:])),(voice,text,rows)
 assert all(0<=r['start']<=r['end']<=len(wave)/32000.+1e-5 for r in rows),(voice,text,rows)
 prompt=engine._semantic_prompt_for_text(text,language);read_as=prompt.language if prompt is not None else language
 expected=set();search=0
 for segment in engine._text_segments(text,read_as):
  at=text.find(segment,search);search=at+len(segment)
  norm=engine.get_phones_and_bert_with_metadata(segment,read_as,engine.model_version,default_lang='ja')['norm_text']
  expected|={r['char_idx'] for r in finalize_timestamps([dict(char_idx=i) for i in range(len(norm))],norm,original_text=segment,char_offset=at)}
 assert {r['char_idx'] for r in rows}==expected,(voice,text,sorted(expected),rows)
 outside=voiced=0
 for first,last in bounds.values():
  piece=wave[first:last];level=np.sqrt((piece[:len(piece)//640*640].reshape(-1,640)**2).mean(axis=1))
  heard=level>.08*level.max();spoken=np.zeros(len(heard),dtype=bool)
  for r in rows:
   if r['phones'] and r['char'] not in SILENT:spoken[max(0,round(r['start']*50-first/640)):max(0,round(r['end']*50-first/640))]=True
  outside+=int((heard&~spoken).sum());voiced+=int(heard.sum())
 # Collapsed characters leave most of the voice outside of them; a sound sits there by nature now and then.
 assert outside<=.35*voiced,(voice,text,outside,voiced,rows)
 return dict(voice=voice,route=route,text=text,characters=len(rows),voice_outside_characters=outside/max(1,voiced),timeline=rows)
for voice,c in voices.items():
 engine=GPTSoVITSInference(str(ROOT/'resources'/c['gpt_path']),str(ROOT/'resources'/c['sovits_path']),str(ROOT/'resources'/c['references']['default']['cache_path']),device=route,cpu_threads=4,pipeline=False)
 try:
  for speed in [1.,1.2]:
   text='早饭想吃什么？我打算煮一碗面，再加一个鸡蛋。'
   reference=None
   for timeline in [False,True]:
    torch.manual_seed(42);started=time.perf_counter();packets=[];rows=[];cursor=0;pre_eos=False;first=None
    for event in LatentStream(engine,seed=9000).stream(text,language='zh',speed=speed,timeline=timeline):
     if event['type']!='chunk':continue
     assert event['start_sample']==cursor
     audio=event['pcm'];assert len(audio)>0 and np.isfinite(audio).all()
     if first is None:first=time.perf_counter()-started
     cursor+=len(audio);packets.append(audio);rows+=event['timestamps'];pre_eos|=not event.get('gpt_finished',True)
    wave=np.concatenate(packets)
    if reference is None:reference=wave
    else:
     np.testing.assert_array_equal(wave,reference)
     assert rows
     assert all(0<=r['start']<=r['end']<=cursor/32000.+1e-5 for r in rows)
     assert len({r['char_idx'] for r in rows})==len(rows)
     sf.write(out/f'{voice}-{route}-{speed}.wav',wave,32000)
     reports.append(dict(voice=voice,route=route,speed=speed,seconds=time.perf_counter()-started,first_chunk=first,duration=cursor/32000.,chunks=len(packets),pre_eos=pre_eos,timeline=rows,audio_parity=True))
     (out/'result.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),'utf-8')
  for language,text in LINES:
   reports.append(check_characters(engine,voice,language,text))
   (out/'result.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),'utf-8')
  assert next(engine.vq_model.flow.parameters()).dtype==next(engine.vq_model.enc_p.parameters()).dtype
 finally:engine.close()
print('PASSED',len(reports))
