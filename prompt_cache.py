"""Prompt-side prebuilt cache for the interactive PyTorch TTS build.

Reference-side tensors (`prompt_semantic`, `refer_spec`, `sv_emb`,
`phones1`/`bert1`/...) only depend on the reference audio + reference text,
both of which are constants per (avatar, reference). We precompute them
offline (via a separate resource-preparation workflow) so the runtime can:

  - Skip librosa.load + ssl + vq.extract_latent + spectrogram_torch +
    sv_model.compute_embedding3 + clean_text on every synthesize call.
  - Avoid loading `ssl_model` / `sv_model` / `chinese-hubert-base` weights
    into the runtime distribution at all.

This module owns serialization. Building lives in the separate preparation
environment. Consumption lives in `GPTSoVITSInference.__init__` /
`.synthesize`.

The runtime uses reference spectrograms and speaker embeddings, independent
of the semantic checkpoint. Legacy semantic tensors remain in cache version 1
for asset compatibility; reference-free inference does not consume them.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch


# Bump when the on-disk format changes in a backwards-incompatible way.
CACHE_VERSION = 1


@dataclass
class PromptCacheMetadata:
    """Provenance information stored next to the tensors. Diagnostic only."""

    ref_audio_filename: str
    ref_text: str
    ref_lang: str
    is_half: bool
    sample_rate: int
    sovits_version: str
    generated_at: str
    notes: str = ""


@dataclass
class PromptCache:
    """In-memory prompt-side cache, ready to be plugged into synthesize."""

    prompt_semantic: torch.Tensor      # Legacy semantic prompt, not used for generation
    refer_spec: torch.Tensor           # (1, n_freq, ~100) float — fed to sovits decode
    sv_emb: torch.Tensor               # (1, ~256) float — fed to sovits decode

    phones1: List[int]
    bert1: torch.Tensor                # Legacy reference text features
    norm_text1: str
    word2ph1: List[int]
    phone_symbols1: List[str]

    metadata: PromptCacheMetadata


def save_prompt_cache(cache: PromptCache, path: str) -> None:
    """Serialize via torch.save with tensors detached + on CPU so files are
    portable across devices/precision modes (caller chooses runtime device
    when loading)."""

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "tensors": {
            "prompt_semantic": cache.prompt_semantic.detach().cpu(),
            "refer_spec": cache.refer_spec.detach().cpu(),
            "sv_emb": cache.sv_emb.detach().cpu(),
            "bert1": cache.bert1.detach().cpu(),
        },
        "phones1": list(cache.phones1),
        "norm_text1": str(cache.norm_text1),
        "word2ph1": list(cache.word2ph1),
        "phone_symbols1": list(cache.phone_symbols1),
        "metadata": asdict(cache.metadata),
    }
    torch.save(payload, path)


def load_prompt_cache(
    path: str,
    expected_is_half: Optional[bool] = None,
    device: Any = "cuda",
) -> PromptCache:
    """Load a cache file and move tensors to ``device``. dtype is automatically
    cast to match the runtime precision when possible."""

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)

    version = payload.get("version")
    if version != CACHE_VERSION:
        raise ValueError(
            f"Prompt cache version mismatch at {path}: cached={version}, "
            f"expected={CACHE_VERSION}. Regenerate the cache with the separate "
            "resource-preparation workflow."
        )

    raw_meta = dict(payload["metadata"])
    valid_fields = {f.name for f in PromptCacheMetadata.__dataclass_fields__.values()}
    extra_keys = [k for k in raw_meta.keys() if k not in valid_fields]
    if extra_keys:
        for k in extra_keys:
            raw_meta.pop(k, None)
    metadata = PromptCacheMetadata(**raw_meta)

    target_float = (
        torch.float16 if (expected_is_half if expected_is_half is not None else metadata.is_half)
        else torch.float32
    )

    def _move(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.to(device)

    tensors = payload["tensors"]
    return PromptCache(
        prompt_semantic=tensors["prompt_semantic"].cpu(),
        refer_spec=_move(tensors["refer_spec"], target_float),
        sv_emb=_move(tensors["sv_emb"], target_float),
        bert1=tensors["bert1"].to(device="cpu", dtype=target_float),
        phones1=list(payload["phones1"]),
        norm_text1=str(payload["norm_text1"]),
        word2ph1=list(payload["word2ph1"]),
        phone_symbols1=list(payload["phone_symbols1"]),
        metadata=metadata,
    )


def inspect_prompt_cache(path: str) -> Dict[str, Any]:
    """Read-only header inspection for diagnostics."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "version": payload.get("version"),
        "metadata": payload.get("metadata"),
        "tensor_shapes": {
            name: list(tensor.shape) for name, tensor in payload.get("tensors", {}).items()
        },
        "tensor_dtypes": {
            name: str(tensor.dtype) for name, tensor in payload.get("tensors", {}).items()
        },
        "phones1_len": len(payload.get("phones1", [])),
    }
