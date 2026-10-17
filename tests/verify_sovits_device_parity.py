"""Real character weights: identical acoustic inputs must preserve CPU voice conditioning."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from run_inference import GPTSoVITSInference

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
import torch_directml
device = torch_directml.device()
voices = json.loads((ROOT / 'resources/voices.json').read_text('utf-8'))
results = []
for voice, config in voices.items():
    engine = GPTSoVITSInference(
        str(ROOT / 'resources' / config['gpt_path']),
        str(ROOT / 'resources' / config['sovits_path']),
        str(ROOT / 'resources' / config['references']['default']['cache_path']),
        device='cpu', cpu_threads=4, pipeline=False)
    try:
        torch.manual_seed(42)
        semantic = engine._generate_segment_semantic('我打算煮一碗面，再加一个鸡蛋。', 'zh', 'ja', 15, 1., 1.)
        codes = semantic['pred_semantic']
        phones = torch.tensor([semantic['phones2']], dtype=torch.long)
        cache = engine.prompt_cache
        with torch.no_grad():
            expected = engine.vq_model.decode(codes, phones, [cache.refer_spec], noise_scale=0,
                                               sv_emb=[cache.sv_emb]).cpu().numpy().copy()
            engine.vq_model.to(device)
            actual = engine.vq_model.decode(codes.to(device), phones.to(device), [cache.refer_spec.to(device)],
                                             noise_scale=0, sv_emb=[cache.sv_emb.to(device)]).cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-4)
        results.append(dict(voice=voice, max_error=float(np.abs(actual-expected).max()), samples=int(actual.size)))
    finally:
        engine.close()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2), 'utf-8')
print('PASSED:', json.dumps(results))
