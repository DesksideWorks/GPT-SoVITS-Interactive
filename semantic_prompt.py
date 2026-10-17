"""Validated, CPU-resident semantic conditioning: one reviewed reference per language and character."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PROMPT_DIR = Path(__file__).resolve().parent / 'resources/semantic_prompts'
LANGUAGES = ('zh', 'ja', 'en')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _checkpoint_hash(path, size, modified_ns):
    return file_sha256(path)


def _weights_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _checkpoint_hash(str(path), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class SemanticPrompt:
    identifier: str
    phones: torch.Tensor
    semantic: torch.Tensor
    bert: torch.Tensor
    language: str = ''

    @property
    def phone_count(self):
        return self.phones.numel()

    @property
    def token_count(self):
        return self.semantic.numel()


def read_manifest(path):
    manifest = json.loads(Path(path).read_text('utf-8'))
    filename = manifest.get('file', '')
    if (manifest.get('schema') != 1 or manifest.get('language') not in LANGUAGES or
            manifest.get('semantic_frame_rate') != 25 or
            not filename or Path(filename).name != filename or '/' in filename or '\\' in filename or
            not filename.endswith('.npz')):
        raise ValueError('Invalid semantic prompt manifest: ' + str(path))
    return manifest


@lru_cache(maxsize=16)
def _load_asset(path, manifest_size, manifest_modified_ns, asset_size, asset_modified_ns):
    manifest = read_manifest(path)
    asset = Path(path).parent / manifest['file']
    if file_sha256(asset) != manifest['sha256']:
        raise ValueError('Semantic prompt checksum mismatch: ' + str(asset))
    with np.load(asset, allow_pickle=False) as data:
        phones, semantic, bert = (data[key] for key in ('phones', 'semantic', 'bert'))
    if (phones.dtype != np.int64 or phones.ndim != 1 or not 0 < phones.size <= 180 or
            semantic.dtype != np.int64 or semantic.ndim != 1 or not 0 < semantic.size <= 256 or
            bert.dtype != np.float32 or bert.shape != (1024, phones.size) or
            not np.isfinite(bert).all() or not ((0 <= phones) & (phones < 732)).all() or
            not ((0 <= semantic) & (semantic < 1024)).all()):
        raise ValueError('Invalid semantic prompt tensor contract: ' + str(asset))
    if phones.size != manifest['phone_count'] or semantic.size != manifest['token_count']:
        raise ValueError('Semantic prompt dimensions differ from its manifest: ' + str(asset))
    return SemanticPrompt(manifest['id'], torch.from_numpy(phones.copy()),
                          torch.from_numpy(semantic.copy()), torch.from_numpy(bert.copy()), manifest['language'])


def load_for_checkpoint(checkpoint, directory=PROMPT_DIR, sovits_checkpoint=None):
    """Prompts bound to these GPT weights, keyed by language.

    A manifest carrying ``sovits_sha256`` belongs to the character with those SoVITS weights;
    the others are shared by every character on this GPT.
    Unknown GPT weights retain their original behavior; corrupt assets fail closed.
    """
    checkpoint_hash = _weights_hash(checkpoint)
    sovits_hash = None
    prompts = {}
    for path in sorted(Path(directory).resolve().glob('*.json')):
        manifest = read_manifest(path)
        if manifest['gpt_sha256'] != checkpoint_hash:
            continue
        owner = manifest.get('sovits_sha256')
        if owner:
            if sovits_checkpoint is None:
                continue
            sovits_hash = sovits_hash or _weights_hash(sovits_checkpoint)
            if owner != sovits_hash:
                continue
        language = manifest['language']
        if language in prompts:
            raise ValueError(f'More than one {language} semantic prompt applies to these weights: '
                             f'{prompts[language].identifier}, {manifest["id"]}')
        ms, ds = path.stat(), (path.parent / manifest['file']).stat()
        prompts[language] = _load_asset(str(path), ms.st_size, ms.st_mtime_ns, ds.st_size, ds.st_mtime_ns)
    return prompts


def prepare_inputs(meta, device, dtype, prompt=None):
    """Only GPT sees the prefix; acoustic decoding still receives target phones."""
    phones = torch.tensor(meta['phones'], dtype=torch.long, device=device)
    bert = meta['bert'].to(device=device, dtype=dtype)
    semantic = None
    if prompt is not None:
        phones = torch.cat((prompt.phones.to(device), phones))
        bert = torch.cat((prompt.bert.to(device=device, dtype=dtype), bert), dim=1)
        semantic = prompt.semantic.to(device).unsqueeze(0)
    phones = phones.unsqueeze(0)
    return phones, torch.tensor([phones.shape[-1]], device=device), semantic, bert.unsqueeze(0)
