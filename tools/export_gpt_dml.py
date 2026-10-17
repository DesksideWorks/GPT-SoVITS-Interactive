"""Build and numerically check versioned GPT assets; run only during asset preparation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import onnx
import onnxruntime as ort
from semantic_backend import TorchGpt
from runtime_backend import resolve_backend
from tools.gpt_onnx_graph import PrefillBucket, Step


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.gpt.with_suffix('.ort-dml')
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    backend = TorchGpt(str(args.gpt), resolve_backend('cpu'))
    model = backend.model
    pre = PrefillBucket(model, 512).eval()
    step = Step(model, 512).eval()
    n = 20
    inputs = (torch.arange(256).remainder(100).reshape(1,256).long(), torch.zeros(1,1024,256),
              torch.where(torch.arange(256)>=n, -10000., 0.).reshape(1,1,1,-1), torch.tensor([n-1]))
    with torch.no_grad():
        logits,k,v = pre(*inputs)
        step_inputs = (torch.tensor([[54]]),torch.tensor([n]),torch.tensor([0]),
                       torch.where(torch.arange(512)>n,-10000.,0.).reshape(1,1,1,-1),k,v)
        expected = step(*step_inputs)
        for name,graph,values,names in (
            ('prefill-p256.onnx',pre,inputs,['phones','bert','attention_bias','last_position']),
            ('step.onnx',step,step_inputs,['token','position','audio_position','attention_bias','cache_k','cache_v'])):
            path = destination/name
            torch.onnx.export(graph,values,str(path),input_names=names,output_names=['logits','keys','values'],opset_version=17)
            onnx.checker.check_model(str(path))
            options = ort.SessionOptions();options.intra_op_num_threads=4
            session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            actual = session.run(None,{key:value.numpy() for key,value in zip(names,values)})
            reference = (logits,k,v) if name.startswith('prefill') else expected
            for a,b in zip(actual,reference):
                np.testing.assert_allclose(a,b.numpy(),rtol=3e-4,atol=1e-3)
    manifest = dict(schema=1,gpt_sha256=digest(args.gpt),config=backend.config,capacity=512,prefill_size=256,
                    files={name:digest(destination/name) for name in ('prefill-p256.onnx','step.onnx')})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    print(f'Export and CPU numerical verification passed: {destination}')


if __name__ == '__main__':
    main()
