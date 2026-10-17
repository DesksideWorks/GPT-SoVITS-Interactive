"""Real-weight semantic conditioning, stream parity, language and lifecycle checks."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
import torch
from run_inference import GPTSoVITSInference
from streaming_latents import LatentStream

TEXTS=dict(zh='冰箱里还有鸡蛋和青菜，晚上我们可以一起做饭。',
           ja='冷蔵庫に卵と野菜があるから、今夜は一緒に料理しようよ。',
           en='There are eggs and vegetables in the fridge, so we can cook together tonight.')
LONG_TEXTS=dict(zh='今天的天气真不错，我想出去散散步，顺便买点新鲜蔬菜，回来以后再一起准备晚饭，你觉得怎么样？我们还可以买一些水果，明天早上一起吃。',
                ja='今日はとても天気がいいから、少し散歩に出かけて、ついでに新鮮な野菜を買って、帰ってから一緒に晩ごはんを作ろうと思うんだけど、どうかな？果物も買って、明日の朝に一緒に食べようね。',
                en='The weather is lovely today, so I would like to go for a walk, pick up some fresh vegetables on the way, and then cook dinner together when we get back, what do you think? We could buy some fruit too, and have it for breakfast tomorrow.')


def stream(engine,text,language='zh',seed=1,timeline=True,speed=1.):
    torch.manual_seed(seed);packets=[];rows=[];cursor=0
    for event in LatentStream(engine,seed=9000).stream(text,language=language,timeline=timeline,speed=speed):
        if event['type']!='chunk':continue
        assert event['start_sample']==cursor
        cursor+=len(event['pcm']);packets.append(event['pcm']);rows+=event['timestamps']
    wave=np.concatenate(packets)
    assert wave.ndim==1 and np.isfinite(wave).all() and len(wave)>0
    if timeline:
        assert rows and all(0<=r['start']<=r['end']<=len(wave)/32000+1e-5 for r in rows)
        assert all(r['char']==text[r['char_idx']] for r in rows)
        assert len({r['char_idx'] for r in rows})==len(rows)
    return wave,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);reports=[]
    voices=json.loads((ROOT/'resources/voices.json').read_text('utf-8'))
    for voice,c in voices.items():
        engine=GPTSoVITSInference(str(ROOT/'resources'/c['gpt_path']),str(ROOT/'resources'/c['sovits_path']),str(ROOT/'resources'/c['references']['default']['cache_path']),device=a.device,cpu_threads=4,pipeline=False)
        try:
            prompts=engine._semantic_prompts
            assert prompts,'No semantic prompt is bound to these GPT weights'
            # Each language reaches GPT with its own prefix, or with the target alone when it has none.
            for language,text in TEXTS.items():
                prompt=engine._semantic_prompt_for_text(text,language)
                assert prompt is prompts.get(language) and engine._semantic_prompt_for_text(text,'auto') is prompt
                meta=engine.get_phones_and_bert_with_metadata(text,language,engine.model_version)
                x,_,prefix,bert=engine._gpt_inputs(meta,prompt)
                if prompt is None:
                    assert prefix is None and x[0].tolist()==meta['phones']
                    torch.testing.assert_close(bert[0],meta['bert'].to(bert),rtol=0,atol=0)
                else:
                    assert prompt.phones.device.type=='cpu' and prefix[0].tolist()==prompt.semantic.tolist()
                    assert x[0].tolist()==prompt.phones.tolist()+meta['phones']
            # A mixed request is not conditioned on any single language.
            mixed='今天喝 coffee。明天说ありがとう。'
            assert engine._semantic_prompt_for_text(mixed,'auto') is None
            meta=engine.get_phones_and_bert_with_metadata(mixed,'auto',engine.model_version)
            x,_,prefix,bert=engine._gpt_inputs(meta,None)
            assert prefix is None and x[0].tolist()==meta['phones']
            durations={}
            for language,prompt in prompts.items():
                text=TEXTS[language]
                torch.manual_seed(1)
                full,timestamps=engine.synthesize(text,text_lang=language,top_k=15,speed=1.,return_timeline=True,pipeline=False)
                assert engine.last_metrics['semantic_prompt_id']==prompt.identifier
                assert timestamps and 2<len(full)/32000<8
                streamed,rows=stream(engine,text,language)
                assert len(full)==len(streamed)
                without,_=stream(engine,text,language,timeline=False)
                np.testing.assert_array_equal(streamed,without)
                sf.write(a.output/f'{voice}-{language}-full.wav',full,32000,subtype='FLOAT')
                sf.write(a.output/f'{voice}-{language}-stream.wav',streamed,32000,subtype='FLOAT')
                faster,_=stream(engine,text,language,speed=1.2)
                assert len(faster)==int((len(streamed)//640)/1.2)*640
                # Cancellation releases the streaming decoder's temporary FP32 mode.
                dtype=next(engine.vq_model.flow.parameters()).dtype
                gen=LatentStream(engine,seed=9000).stream(text,language=language)
                try:
                    for event in gen:
                        if event['type']=='chunk':break
                finally:gen.close()
                assert next(engine.vq_model.flow.parameters()).dtype==dtype
                engine.hibernate();assert prompt.bert.device.type=='cpu'
                engine.wake();again,_=stream(engine,text,language)
                np.testing.assert_array_equal(again,streamed)
                durations[language]=len(streamed)/32000
                if voice=='tyc':
                    long_text=LONG_TEXTS[language]
                    segments=engine._text_segments(long_text,language)
                    # Sentence splitting only drops the spaces between sentences.
                    assert ''.join(segments).replace(' ','')==long_text.replace(' ','')
                    if a.device=='dml':
                        budget=engine.t2s_model.segment_phone_budget(prompt.phone_count,prompt.token_count)
                        assert len(segments)>1
                        assert all(len(engine.get_phones_and_bert_with_metadata(s,language,engine.model_version)['phones'])<=budget for s in segments)
                    long_wave,long_rows=stream(engine,long_text,language,seed=42)
                    sf.write(a.output/f'long-{language}-stream.wav',long_wave,32000,subtype='FLOAT')
                    (a.output/f'long-{language}-timeline.json').write_text(json.dumps(long_rows,ensure_ascii=False),'utf-8')
            reports.append(dict(voice=voice,device=a.device,prompts={k:v.identifier for k,v in prompts.items()},durations=durations,full_stream_duration_equal=True,timeline_audio_parity=True,cancellation=True,hibernate_wake=True,language_isolation=True))
            (a.output/'result.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),'utf-8')
        finally:engine.close()
    print('PASSED',json.dumps(reports),flush=True)


if __name__=='__main__':main()
