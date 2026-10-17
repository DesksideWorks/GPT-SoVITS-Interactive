"""Prepare a reviewed per-language semantic prefix for bundled inference.

One of three sources produces the same asset (an NPZ plus its JSON manifest):

  --audio         a short native-speaker recording, encoded with HuBERT and the voice's quantizer
  --prompt-cache  a character reference cache, whose tokens were encoded that way when it was built
  --generated     a reviewed semantic generation record from the GPT itself

--character-only binds the prefix to --voice, for a reference in that character's own voice;
without it every character on the GPT shares the prefix.

Run this in the resource-preparation environment. The runtime only reads the result and
never loads HuBERT.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from semantic_prompt import LANGUAGES, file_sha256, load_for_checkpoint

TRAILING_SILENCE_SEC = 0.3
RESOURCES = ROOT / 'resources'


def voice_config(name):
    return json.loads((RESOURCES / 'voices.json').read_text('utf-8'))[name]


def from_generated(args):
    from chinese_frontend import get_bert_feature
    source = json.loads(args.generated.read_text('utf-8'))
    if len(source['segments']) != 1 or source['speed'] != 1 or source.get('error'):
        raise ValueError('Expected one reviewed unmodified semantic segment')
    if not args.expected_gpt_sha256 or file_sha256(args.checkpoint) != args.expected_gpt_sha256.lower():
        raise ValueError('Reference and deployment GPT weights differ')
    segment = source['segments'][0]
    if any(span['language'] != 'zh' for span in segment['spans']):
        raise ValueError('A generated reference must contain only Chinese speech')
    bert = get_bert_feature(segment['norm_text'], segment['word2ph']).float().cpu().numpy()
    provenance = dict(kind='generated-semantic-prefix', reference_id=source['id'],
                      source_sha256=file_sha256(args.generated), seed=source['seed'],
                      sampling=source['params'], pcm_encoded=False)
    return 'zh', segment['phones'], segment['tokens'], bert, segment['text'], segment['norm_text'], provenance


def from_prompt_cache(args):
    payload = torch.load(args.prompt_cache, map_location='cpu', weights_only=False)
    metadata = payload['metadata']
    provenance = dict(kind='character-reference-cache', pcm_encoded=True,
                      cache_sha256=file_sha256(args.prompt_cache),
                      reference_audio=metadata['ref_audio_filename'], cache_generated_at=metadata['generated_at'])
    return (metadata['ref_lang'], payload['phones1'], payload['tensors']['prompt_semantic'].flatten().tolist(),
            payload['tensors']['bert1'].float().numpy(), metadata['ref_text'], payload['norm_text1'], provenance)


def from_audio(args):
    """Upstream reference encoding: 16 kHz, 0.3 s of trailing zeros, raw waveform into HuBERT."""
    import soundfile as sf
    import torchaudio.functional as AF
    from transformers import HubertModel
    from run_inference import GPTSoVITSInference
    if not (args.text and args.language and args.hubert and args.voice):
        raise ValueError('--audio requires --text, --language, --hubert and --voice')
    voice = voice_config(args.voice)
    resources = RESOURCES
    if (resources / voice['gpt_path']).resolve() != args.checkpoint.resolve():
        raise ValueError('The voice must use the GPT weights this prefix is bound to')
    wav, sample_rate = sf.read(args.audio, dtype='float32')
    if wav.ndim != 1:
        raise ValueError('Reference audio must be mono')
    engine = GPTSoVITSInference(str(resources / voice['gpt_path']), str(resources / voice['sovits_path']),
                                str(resources / voice['references'][voice['default_reference_id']]['cache_path']),
                                device=args.device, cpu_threads=4)
    try:
        device = engine.device
        hubert = HubertModel.from_pretrained(str(args.hubert), local_files_only=True).to(device).eval()
        wav16 = AF.resample(torch.from_numpy(wav), sample_rate, 16000).to(device)
        wav16 = torch.cat((wav16, torch.zeros(int(16000 * TRAILING_SILENCE_SEC), device=device)))
        with torch.no_grad():
            features = hubert(wav16.unsqueeze(0))['last_hidden_state'].transpose(1, 2)
            tokens = engine.vq_model.extract_latent(features.to(next(engine.vq_model.parameters()).dtype)).flatten()
        if abs(tokens.numel() / (wav16.numel() / 16000) - 25.0) > 3.0:
            raise ValueError('Encoded reference does not have the 25 Hz semantic frame rate')
        meta = engine.get_phones_and_bert_with_metadata(args.text, args.language, engine.model_version)
        if [span['language'] for span in meta['language_spans']] != [args.language]:
            raise ValueError('Reference text is not read wholly as ' + args.language)
        provenance = dict(kind='encoded-reference-audio', pcm_encoded=True, audio_sha256=file_sha256(args.audio),
                          audio_seconds=round(len(wav) / sample_rate, 3), trailing_silence_sec=TRAILING_SILENCE_SEC,
                          sovits_sha256=file_sha256(resources / voice['sovits_path']))
        return (args.language, meta['phones'], tokens.cpu().tolist(), meta['bert'].float().cpu().numpy(),
                args.text, meta['norm_text'], provenance)
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--audio', type=Path, help='Mono reference recording')
    source.add_argument('--prompt-cache', type=Path, help='Character reference cache (.cache.pt)')
    source.add_argument('--generated', type=Path, help='Reviewed semantic inference JSON')
    parser.add_argument('--id', required=True, help='Asset name, for example en-arctic-clb-v1')
    parser.add_argument('--checkpoint', type=Path, required=True, help='GPT weights the prefix is bound to')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--text', help='Exact transcript of --audio')
    parser.add_argument('--language', choices=LANGUAGES, help='Language of --audio')
    parser.add_argument('--hubert', type=Path, help='chinese-hubert-base directory, for --audio')
    parser.add_argument('--voice', help='Entry of resources/voices.json whose quantizer encodes --audio')
    parser.add_argument('--character-only', action='store_true', help='Only --voice uses this prefix')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected-gpt-sha256', help='Hash recorded when --generated was produced')
    parser.add_argument('--source', help='Where the recording comes from: corpus, speaker and file')
    parser.add_argument('--license', help='License of the recording')
    parser.add_argument('--attribution', help='Notice the license requires')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', args.id):
        raise ValueError('--id may contain lower-case letters, digits and hyphens')
    if not args.generated and not (args.source and args.license):
        raise ValueError('A recorded reference needs --source and --license')
    if args.character_only and not args.voice:
        raise ValueError('--character-only requires --voice')
    sovits = RESOURCES / voice_config(args.voice)['sovits_path'] if args.character_only else None

    build = from_audio if args.audio else from_prompt_cache if args.prompt_cache else from_generated
    language, phones, tokens, bert, text, normalized_text, provenance = build(args)
    for key in ('source', 'license', 'attribution'):
        if getattr(args, key):
            provenance[key] = getattr(args, key)
    args.output.mkdir(parents=True, exist_ok=True)
    asset = args.output / (args.id + '.npz')
    np.savez_compressed(asset, phones=np.asarray(phones, dtype=np.int64),
                        semantic=np.asarray(tokens, dtype=np.int64), bert=bert)
    manifest = dict(schema=1, id=args.id, language=language, semantic_frame_rate=25,
                    gpt_sha256=file_sha256(args.checkpoint),
                    **(dict(sovits_sha256=file_sha256(sovits)) if sovits else {}),
                    file=asset.name, sha256=file_sha256(asset),
                    phone_count=len(phones), token_count=len(tokens),
                    text=text, normalized_text=normalized_text, provenance=provenance)
    (args.output / (args.id + '.json')).write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
    # The runtime loader is the contract: size limits, one prefix per language for each character.
    loaded = load_for_checkpoint(args.checkpoint, args.output, sovits)
    assert loaded[language].identifier == args.id
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
