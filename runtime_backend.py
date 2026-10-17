"""Runtime device selection shared by PyTorch SoVITS and semantic backends."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


DEVICE_CHOICES: Tuple[str, ...] = ("auto", "cpu", "cuda", "dml", "directml")

_DML_SDPA_DISABLED = False
_DML_SDPA_WARNED = False
_DML_SAMPLING_FALLBACK = False
_DML_SAMPLING_WARNED = False
_DML_RANDN_FALLBACK = False
_DML_RANDN_WARNED = False


@dataclass(frozen=True)
class RuntimeBackend:
    requested: str
    kind: str
    torch_device: Any
    label: str
    is_half: bool
    directml_module: Optional[Any] = None

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def is_directml(self) -> bool:
        return self.kind == "dml"

    def synchronize(self) -> None:
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(self.torch_device)

    def empty_cache(self) -> None:
        if self.is_cuda and torch.cuda.is_available():
            with torch.cuda.device(self.torch_device):
                torch.cuda.empty_cache()

    def reset_peak_memory_stats(self) -> None:
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.torch_device)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "kind": self.kind,
            "torch_device": str(self.torch_device),
            "label": self.label,
            "is_half": self.is_half,
        }


def normalize_device_request(requested: str = "auto") -> str:
    value = (requested or "auto").strip().lower()
    if value == "directml":
        value = "dml"
    if value not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported device '{requested}'. Expected one of: {', '.join(DEVICE_CHOICES)}")
    return value


def _directml_device_name(module: Any, index: int = 0) -> str:
    for name in ("device_name", "get_device_name"):
        func = getattr(module, name, None)
        if callable(func):
            try:
                return str(func(index)).rstrip("\x00")
            except TypeError:
                try:
                    return str(func()).rstrip("\x00")
                except Exception:
                    pass
            except Exception:
                pass
    return f"DirectML:{index}"


def _normalize_device_name(value: str) -> str:
    return " ".join((value or "").replace("\x00", "").strip().lower().split())


def is_integrated_graphics_name(value: str) -> bool:
    name = _normalize_device_name(value)
    if not name:
        return False
    if "microsoft basic render" in name or "remote display" in name:
        return True
    intel_markers = (
        "intel(r) hd",
        "intel hd",
        "intel(r) uhd",
        "intel uhd",
        "iris xe",
        "iris plus",
    )
    amd_markers = (
        "radeon(tm) graphics",
        "amd radeon graphics",
        "radeon 660m",
        "radeon 680m",
        "radeon 760m",
        "radeon 780m",
        "radeon 860m",
        "radeon 880m",
        "radeon 890m",
        "radeon vega",
    )
    return any(marker in name for marker in intel_markers + amd_markers)


def _directml_devices(module: Any) -> List[Tuple[int, str]]:
    count_func = getattr(module, "device_count", None)
    try:
        count = max(0, int(count_func())) if callable(count_func) else 1
    except Exception:
        count = 1
    return [(index, _directml_device_name(module, index)) for index in range(count)]


def _matching_device_index(devices: List[Tuple[int, str]], preferred_name: Optional[str]) -> Optional[int]:
    wanted = _normalize_device_name(preferred_name or "")
    if not wanted:
        return None
    for index, name in devices:
        candidate = _normalize_device_name(name)
        if candidate == wanted or candidate in wanted or wanted in candidate:
            return index
    return None


def _create_directml_backend(requested: str, preferred_name: Optional[str] = None) -> RuntimeBackend:
    try:
        torch_directml = importlib.import_module("torch_directml")
    except ImportError as exc:
        raise RuntimeError(
            "DirectML was requested, but torch_directml is not installed in this Python runtime."
        ) from exc

    devices = _directml_devices(torch_directml)
    if not devices:
        raise RuntimeError("DirectML was requested, but no DirectML device is available.")

    selected_index = _matching_device_index(devices, preferred_name)
    if selected_index is None:
        selected_index = next(
            (index for index, name in devices if not is_integrated_graphics_name(name)),
            devices[0][0],
        )

    try:
        device = torch_directml.device(selected_index)
    except Exception as exc:
        raise RuntimeError(f"DirectML device initialization failed: {exc}") from exc

    label = _directml_device_name(torch_directml, selected_index)
    return RuntimeBackend(
        requested=requested,
        kind="dml",
        torch_device=device,
        label=label,
        # SoVITS uses FP32 for torch-directml operator compatibility.
        is_half=False,
        directml_module=torch_directml,
    )


def _create_cuda_backend(requested: str, preferred_name: Optional[str] = None) -> RuntimeBackend:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    count = max(1, int(torch.cuda.device_count()))
    devices = [(index, str(torch.cuda.get_device_name(index)).rstrip("\x00")) for index in range(count)]
    selected_index = _matching_device_index(devices, preferred_name)
    if selected_index is None:
        selected_index = 0
    return RuntimeBackend(
        requested=requested,
        kind="cuda",
        torch_device=f"cuda:{selected_index}",
        label=devices[selected_index][1],
        is_half=True,
    )


def resolve_backend(requested: str = "auto", preferred_name: Optional[str] = None) -> RuntimeBackend:
    normalized = normalize_device_request(requested)

    if normalized == "cpu":
        return RuntimeBackend(requested=requested, kind="cpu", torch_device="cpu", label="CPU", is_half=False)

    if normalized == "cuda":
        return _create_cuda_backend(requested, preferred_name)

    if normalized == "dml":
        return _create_directml_backend(requested, preferred_name)

    if torch.cuda.is_available():
        return resolve_backend("cuda", preferred_name)

    try:
        return _create_directml_backend(requested, preferred_name)
    except RuntimeError:
        return resolve_backend("cpu")


def resolve_device(requested: str = "auto", preferred_name: Optional[str] = None) -> Any:
    return resolve_backend(requested, preferred_name).torch_device


def is_directml_device(device: Any) -> bool:
    return getattr(device, "type", None) == "privateuseone" or str(device).startswith("privateuseone")


def _env_choice(name: str, default: str, allowed: Tuple[str, ...]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in allowed:
        print(f"[runtime_backend] ignoring invalid {name}={value!r}; expected one of {allowed}")
        return default
    return value


def directml_sdpa_mode() -> str:
    return _env_choice("GSV_TTS_DML_SDPA", "auto", ("auto", "on", "off"))


def directml_sampling_mode() -> str:
    return _env_choice("GSV_TTS_DML_SAMPLING", "auto", ("auto", "cpu", "multinomial", "gumbel", "argmax"))


def directml_randn_mode() -> str:
    return _env_choice("GSV_TTS_DML_RANDN", "auto", ("auto", "cpu", "native"))


def should_use_directml_sdpa(device: Any) -> bool:
    if not is_directml_device(device):
        return True
    mode = directml_sdpa_mode()
    if mode == "off":
        return False
    return not _DML_SDPA_DISABLED


def mark_directml_sdpa_failed(exc: BaseException) -> None:
    global _DML_SDPA_DISABLED, _DML_SDPA_WARNED
    if directml_sdpa_mode() == "on":
        raise exc
    _DML_SDPA_DISABLED = True
    if not _DML_SDPA_WARNED:
        print(f"[runtime_backend] DirectML SDPA failed once; falling back to manual attention: {exc}")
        _DML_SDPA_WARNED = True


def should_use_directml_cpu_sampling() -> bool:
    return directml_sampling_mode() == "cpu" or _DML_SAMPLING_FALLBACK


def mark_directml_sampling_failed(exc: BaseException) -> None:
    global _DML_SAMPLING_FALLBACK, _DML_SAMPLING_WARNED
    if directml_sampling_mode() in ("multinomial", "gumbel", "argmax"):
        raise exc
    _DML_SAMPLING_FALLBACK = True
    if not _DML_SAMPLING_WARNED:
        print(f"[runtime_backend] DirectML sampling failed once; falling back to CPU sampling: {exc}")
        _DML_SAMPLING_WARNED = True


def should_use_directml_cpu_randn() -> bool:
    return directml_randn_mode() == "cpu" or _DML_RANDN_FALLBACK


def mark_directml_randn_failed(exc: BaseException) -> None:
    global _DML_RANDN_FALLBACK, _DML_RANDN_WARNED
    if directml_randn_mode() == "native":
        raise exc
    _DML_RANDN_FALLBACK = True
    if not _DML_RANDN_WARNED:
        print(f"[runtime_backend] DirectML randn failed once; falling back to CPU randn: {exc}")
        _DML_RANDN_WARNED = True


def directml_policy_snapshot() -> Dict[str, Any]:
    return {
        "sdpa": {
            "mode": directml_sdpa_mode(),
            "disabled_after_failure": _DML_SDPA_DISABLED,
        },
        "sampling": {
            "mode": directml_sampling_mode(),
            "cpu_fallback_after_failure": _DML_SAMPLING_FALLBACK,
        },
        "randn": {
            "mode": directml_randn_mode(),
            "cpu_fallback_after_failure": _DML_RANDN_FALLBACK,
        },
    }
