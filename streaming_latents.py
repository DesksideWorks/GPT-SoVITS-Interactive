"""Pre-EOS PCM stream with immutable acoustic frames.

Global attention is evaluated on available semantic prefixes. Only future
acoustic frames may change. Already committed z_p is immutable; finite-support
flow/vocoder decode with context and crop, never crossfade unrelated PCM.
Timeline is a side channel and cannot alter latent/audio generation.
"""
import math,time
import numpy as np
import torch
from torch.nn import functional as F
from timeline import StreamingTimeline

class LatentStream:
    def __init__(self,engine,token_chunk=12,lookahead_tokens=16,context_frames=64,seed=None):
        self.engine=engine;self.model=engine.vq_model
        self.token_chunk=token_chunk;self.lookahead=lookahead_tokens*2;self.context=context_frames;self.seed=seed
        assert engine.sample_rate==32000 and self.model.semantic_frame_rate=='25hz'
        assert engine.model_version=='v2ProPlus'
        assert self.model.flow.n_flows==4 and self.model.flow.kernel_size==5 and self.model.flow.n_layers==4
        assert list(self.model.upsample_rates)==[10,8,2,2,2]
        assert list(self.model.resblock_kernel_sizes)==[3,7,11]
        assert [list(d) for d in self.model.resblock_dilation_sizes]==[[1,3,5]]*3
        # Flow radius 32 frames, vocoder conservatively <32 additional frames.
        # Only the local acoustic decoder uses FP32, limiting cropped-kernel rounding.
        assert self.context>=64
        self.decode_dtype = next(self.model.flow.parameters()).dtype

    @torch.no_grad()
    def condition(self):
        g=self.model;cache=self.engine.prompt_cache
        ref=cache.refer_spec.to(self.engine.device);sv=cache.sv_emb.to(self.engine.device)
        mask=ref.new_ones((1,1,ref.shape[-1]));ge=g.ref_enc(ref[:,:704]*mask,mask)
        if g.is_v2pro:
            ge=g.prelu(ge+g.sv_emb(sv).unsqueeze(-1));gep=g.ge_to512(ge.transpose(2,1)).transpose(2,1)
        else:gep=ge
        return ge,gep

    @torch.no_grad()
    def acoustic(self,codes,phones,gep,speed):
        g=self.model;quantized=g.quantizer.decode(codes.unsqueeze(0))
        quantized=F.interpolate(quantized,scale_factor=2.,mode='nearest')
        length=torch.tensor([quantized.shape[-1]],device=codes.device)
        _,mean,logs,mask,_,_=g.enc_p(quantized,length,phones,torch.tensor([phones.shape[-1]],device=phones.device),gep,speed)
        attn=g.enc_p.mrte.cross_attention.attn
        return mean,logs,attn

    @torch.no_grad()
    def decode(self,zp,ge):
        mask=zp.new_ones((1,1,zp.shape[-1]));z=self.model.flow(zp.to(self.decode_dtype),mask.to(self.decode_dtype),g=ge.to(self.decode_dtype),reverse=True)
        return self.model.dec(z,g=ge.to(self.decode_dtype))[0,0]

    def stream(self,*args,**kwargs):
        dtype=next(self.model.flow.parameters()).dtype
        self.model.flow.float();self.model.dec.float();self.decode_dtype=torch.float32
        try:
            yield from self._stream(*args,**kwargs)
        finally:
            self.model.flow.to(dtype=dtype);self.model.dec.to(dtype=dtype)

    def _stream(self,text,language='auto',timeline=True,pause=.3,noise_scale=.35,speed=1.,top_k=15,top_p=1.,temperature=1.):
        engine=self.engine;offset=0;source_offset=0;seq=0;start=time.perf_counter()
        yield dict(type='start',schema='gsv-latent-stream-v1',sample_rate=32000,channels=1,format='float32-le',timeline=timeline,text_index_unit='unicode_codepoint')
        ge,gep=self.condition()
        semantic_prompt=engine._semantic_prompt_for_text(text,language)
        # Every fragment of a prefixed request is read in the prefix's language, not detected again.
        if semantic_prompt is not None:language=semantic_prompt.language
        segments=engine._text_segments(text,language)
        for si,segment in enumerate(segments):
            char_offset=text.find(segment,source_offset);assert char_offset>=0;source_offset=char_offset+len(segment)
            meta=engine.get_phones_and_bert_with_metadata(segment,language,engine.model_version,default_lang='ja')
            x,x_length,prompt,bert=engine._gpt_inputs(meta,semantic_prompt)
            phones=torch.tensor([meta['phones']],dtype=torch.long,device=engine.device)
            # Local noise RNG cannot change GPT sampling. Frame-major allocation
            # makes the same absolute frame retain the same noise for every prefix.
            generator=torch.Generator(device='cpu')
            generator.manual_seed(self.seed+si) if self.seed is not None else generator.seed()
            noise=torch.randn((8192,self.model.enc_p.out_channels),generator=generator).T.unsqueeze(0)
            tokens=[];commits=[];committed=0;emitted=0;attn=None
            aligner=StreamingTimeline(meta['word2ph'],meta['phone_symbols'],meta['norm_text'],speed,offset/32000.,segment,char_offset) if timeline else None
            token_iter=engine.t2s_model.model.infer_panel_naive(x,x_length,prompt,bert,
                top_k=top_k,top_p=top_p,temperature=temperature,early_stop_num=1500,streaming_mode=True,chunk_length=self.token_chunk)
            for token_piece,final in token_iter:
                if token_piece is not None and token_piece.numel():tokens.append(token_piece.detach().to(engine.device))
                codes=torch.cat(tokens,dim=1);raw_available=2*codes.shape[-1];available=int(raw_available/speed)
                commit_to=available if final else max(0,available-math.ceil(self.lookahead/speed))
                if commit_to>committed:
                    mean,logs,attn=self.acoustic(codes,phones,gep,speed)
                    assert available<noise.shape[-1]
                    m=mean[:,:,committed:commit_to].float();l=logs[:,:,committed:commit_to].float()
                    new=m+noise[:,:,committed:commit_to].to(m)*torch.exp(l)*noise_scale
                    commits.append(new);assert torch.isfinite(new).all()
                    committed=commit_to
                safe_end=committed if final else max(0,committed-self.context)
                if safe_end<=emitted:continue
                all_z=torch.cat(commits,dim=-1)
                left=max(0,emitted-self.context);right=committed if final else min(committed,safe_end+self.context)
                wave=self.decode(all_z[:,:,left:right],ge)
                pcm=(wave[(emitted-left)*640:(safe_end-left)*640]*.9).float().cpu().numpy().copy()
                assert len(pcm)==(safe_end-emitted)*640 and np.isfinite(pcm).all() and np.max(abs(pcm))<=.901
                # A character goes out with the audio that holds its start, placed by
                # the newest alignment. Timing is sample-based, never stretched to text end.
                pending=[dict(r,approximate=True) for r in aligner.collect(attn,safe_end,final)] if timeline else []
                event=dict(type='chunk',seq=seq,start_sample=offset+emitted*640,sample_count=len(pcm),pcm=pcm,
                           timestamps=pending,segment=si,gpt_tokens=int(codes.shape[-1]),gpt_finished=final,elapsed=time.perf_counter()-start)
                
                emitted=safe_end;seq+=1
                yield event
            assert emitted==committed==available
            offset+=emitted*640
            if si<len(segments)-1 and pause>0:
                pcm=np.zeros(int(pause*32000),dtype='float32')
                yield dict(type='chunk',seq=seq,start_sample=offset,sample_count=len(pcm),pcm=pcm,timestamps=[],segment=si,pause=True,gpt_finished=True,elapsed=time.perf_counter()-start)
                offset+=len(pcm);seq+=1
        yield dict(type='done',chunks=seq,total_samples=offset,duration=offset/32000,elapsed=time.perf_counter()-start,
                   reached_token_limit=False)
