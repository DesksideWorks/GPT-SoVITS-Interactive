"""Verify the restored offline payload and pinned runtime dependency imports."""
import ast
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    manifest=json.loads((ROOT/'chinese-runtime-manifest.json').read_text('utf-8'))
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT) and digest(path)==row['sha256'],row['path']
        if path.suffix=='.py':ast.parse(path.read_text('utf-8'))
    imports={'cn2an':'cn2an','pypinyin':'pypinyin','jieba':'jieba','jieba_fast':'jieba-fast',
        'split_lang':'split-lang','fast_langdetect':'fast-langdetect',
        'onnxruntime':'onnxruntime', 'opencc':'opencc-python-reimplemented'}
    # The Python module is named ``onnxruntime`` for both CPU and DirectML
    # wheels; distribution metadata differs by environment.
    try:
        importlib.metadata.version('onnxruntime-directml')
        ort_distribution = 'onnxruntime-directml'
    except importlib.metadata.PackageNotFoundError:
        ort_distribution = 'onnxruntime'
    imports['onnxruntime'] = ort_distribution
    versions={}
    for module,distribution in imports.items():
        importlib.import_module(module)
        versions[distribution]=importlib.metadata.version(distribution)
    import torch
    config=json.loads((ROOT/'resources/voices.json').read_text('utf-8'))
    assert config['tyc']['gpt_path']==config['ami']['gpt_path']=='models/ami/gpt.ckpt'
    print(json.dumps(dict(passed=True,files=len(manifest['files']),python=sys.executable,
        torch=torch.__version__,packages=versions)))


if __name__=='__main__':main()
