"""Build a character's acoustic reference cache (.cache.pt) from one reference recording.

The runtime never reads the recording itself: every synthesis takes the reference spectrogram
and the speaker embedding from this cache. The cache also keeps the recording's semantic tokens
and text features, which `tools/build_semantic_prompt.py --prompt-cache` can turn into a
semantic prefix for the character.

Needs two upstream GPT-SoVITS pretrained models that the runtime does not ship:
  --hubert  the chinese-hubert-base directory, for the semantic tokens
  --sv      sv/pretrained_eres2netv2w24s4ep4.ckpt, for the speaker embedding

Add the character to resources/voices.json first; the cache is written to the reference's
cache_path there unless --output says otherwise. Use a clean 3-10 second mono recording and its
exact transcript. After writing, the cache is loaded by the runtime engine, which reads a short
sentence with it; --sample keeps that audio for listening.

Run this in the resource-preparation environment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'GPT_SoVITS'))
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from prompt_cache import PromptCache, PromptCacheMetadata, save_prompt_cache
from run_inference import DictToAttrRecursive, GPTSoVITSInference
from semantic_prompt import LANGUAGES

RESOURCES = ROOT / 'resources'
TRAILING_SILENCE_SEC = 0.3
SAMPLES = {
    'zh': '你好，很高兴认识你。',
    'ja': 'こんにちは、よろしくお願いします。',
    'en': 'Hello, nice to meet you.',
}


class TextFrontend:
    """The runtime's own text processing for the reference transcript, without any model weights."""
    gpt_device = 'cpu'
    gpt_is_half = False
    empty_bert = GPTSoVITSInference.empty_bert
    _gpt_dtype = GPTSoVITSInference._gpt_dtype
    get_phones_and_bert_with_metadata = GPTSoVITSInference.get_phones_and_bert_with_metadata


def load_sovits(path, device):
    """The character's SoVITS as the runtime builds it, in float32."""
    from GPT_SoVITS.module.models import SynthesizerTrn
    from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
    checkpoint = load_sovits_new(str(path))
    hps = DictToAttrRecursive(checkpoint['config'])
    hps.model.semantic_frame_rate = '25hz'
    if 'version' in checkpoint['config'].get('model', {}):
        version = checkpoint['config']['model']['version']
    elif 'sv_emb.weight' in checkpoint['weight']:
        version = 'v2Pro'
    else:
        _, version, _ = get_sovits_version_from_path_fast(str(path))
    hps.model.version = version
    model = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length,
                           n_speakers=hps.data.n_speakers, **hps.model)
    model.load_state_dict(checkpoint['weight'], strict=False)
    if not model.is_v2pro:
        raise ValueError(f'SoVITS {version} has no speaker embedding input; expected v2Pro or v2ProPlus')
    return model.to(device).eval(), hps


def encode_reference(wav, sample_rate, sovits, hps, args, device):
    """Semantic tokens, reference spectrogram and speaker embedding, as upstream GPT-SoVITS computes them."""
    from GPT_SoVITS.module.mel_processing import spectrogram_torch
    from transformers import HubertModel
    wav = torch.from_numpy(wav)
    with torch.no_grad():
        # Same encoding as build_semantic_prompt.py --audio: 16 kHz, 0.3 s of trailing zeros, raw waveform.
        wav16 = AF.resample(wav, sample_rate, 16000).to(device)
        wav16 = torch.cat((wav16, torch.zeros(int(16000 * TRAILING_SILENCE_SEC), device=device)))
        hubert = HubertModel.from_pretrained(str(args.hubert), local_files_only=True).to(device).eval()
        features = hubert(wav16.unsqueeze(0))['last_hidden_state'].transpose(1, 2)
        tokens = sovits.extract_latent(features)[0, 0].unsqueeze(0)
        del hubert
        if abs(tokens.shape[1] / (wav16.numel() / 16000) - 25.0) > 3.0:
            raise ValueError('Encoded reference does not have the 25 Hz semantic frame rate')

        audio = AF.resample(wav, sample_rate, hps.data.sampling_rate).unsqueeze(0).to(device)
        peak = audio.abs().max()
        if peak > 1:
            audio = audio / min(2, peak)
        spec = spectrogram_torch(audio, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                                 hps.data.win_length, center=False)

        os.environ['SV_MODEL_PATH'] = str(args.sv)
        from sv import SV
        sv_emb = SV(device, False).compute_embedding3(AF.resample(audio, hps.data.sampling_rate, 16000))
    return tokens.cpu(), spec.cpu(), sv_emb.cpu()


def check_with_runtime(voice, cache_path, args):
    """Loads the new cache the way the service does and reads one sentence with it."""
    engine = GPTSoVITSInference(str(RESOURCES / voice['gpt_path']), str(RESOURCES / voice['sovits_path']),
                                str(cache_path), device=args.device)
    try:
        audio, _ = engine.synthesize(text=SAMPLES[args.language], text_lang=args.language)
        seconds = len(audio) / engine.sample_rate
        if not np.isfinite(audio).all() or not 0.5 <= seconds <= 10:
            raise ValueError(f'The runtime produced unusable audio with this cache ({seconds:.2f} s)')
        if args.sample:
            sf.write(args.sample, audio, engine.sample_rate)
        return seconds
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--voice', required=True, help='Entry of resources/voices.json')
    parser.add_argument('--reference', help="Reference id under the entry; defaults to its default_reference_id")
    parser.add_argument('--audio', type=Path, required=True, help='Reference recording, 3-10 seconds')
    parser.add_argument('--text', required=True, help='Exact transcript of --audio')
    parser.add_argument('--language', required=True, choices=LANGUAGES, help='Language of --audio')
    parser.add_argument('--hubert', type=Path, required=True, help='chinese-hubert-base directory')
    parser.add_argument('--sv', type=Path, required=True, help='pretrained_eres2netv2w24s4ep4.ckpt')
    parser.add_argument('--output', type=Path, help="Defaults to the reference's cache_path in voices.json")
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing cache')
    parser.add_argument('--sample', type=Path, help='Also write the check sentence to this WAV file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=('cuda', 'cpu'))
    args = parser.parse_args()

    voice = json.loads((RESOURCES / 'voices.json').read_text('utf-8'))[args.voice]
    reference = args.reference or voice['default_reference_id']
    output = args.output or RESOURCES / voice['references'][reference]['cache_path']
    if output.exists() and not args.overwrite:
        raise SystemExit(f'{output} exists; pass --overwrite to replace it')
    if not args.hubert.is_dir() or not args.sv.is_file():
        raise SystemExit('--hubert must be the chinese-hubert-base directory and --sv the checkpoint file')

    wav, sample_rate = sf.read(args.audio, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    seconds = len(wav) / sample_rate
    if not 3 <= seconds <= 10:
        print(f'[warning] the reference is {seconds:.2f} s; GPT-SoVITS expects 3-10 s', flush=True)

    sovits, hps = load_sovits(RESOURCES / voice['sovits_path'], args.device)
    version = str(hps.model.version)
    text = TextFrontend().get_phones_and_bert_with_metadata(args.text, args.language, version)
    if [span['language'] for span in text['language_spans']] != [args.language]:
        raise ValueError('The transcript is not read wholly as ' + args.language)
    tokens, spec, sv_emb = encode_reference(wav, sample_rate, sovits, hps, args, args.device)
    del sovits
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    cache = PromptCache(
        prompt_semantic=tokens, refer_spec=spec, sv_emb=sv_emb,
        phones1=text['phones'], bert1=text['bert'].cpu(), norm_text1=text['norm_text'],
        word2ph1=text['word2ph'], phone_symbols1=text['phone_symbols'],
        metadata=PromptCacheMetadata(
            ref_audio_filename=args.audio.name, ref_text=args.text, ref_lang=args.language, is_half=False,
            sample_rate=int(hps.data.sampling_rate), sovits_version=version,
            generated_at=datetime.now(timezone.utc).isoformat(), notes='tools/build_prompt_cache.py'))
    pending = output.with_name(output.name + '.new')
    save_prompt_cache(cache, str(pending))
    try:
        sample_seconds = check_with_runtime(voice, pending, args)
    except BaseException:
        pending.unlink()
        raise
    os.replace(pending, output)
    print(json.dumps(dict(cache=str(output), reference_seconds=round(seconds, 3), tokens=tokens.shape[1],
                          phones=len(text['phones']), check_seconds=round(sample_seconds, 3)),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
