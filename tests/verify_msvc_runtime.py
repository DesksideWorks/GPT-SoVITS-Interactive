"""Check the real loaded VC DLL paths, not just presence beside python.exe.

Run this script with the specific bundled Python being checked. No installation,
registry edit, system DLL replacement or host application startup is performed.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import traceback


def loaded_dll_path(name):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint
    handle = kernel.GetModuleHandleW(name)
    if not handle:
        return None
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "dml"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(sys.executable).resolve().parent
    manifest = json.loads((root.parent.parent / "msvc-runtime-manifest.json").read_text(encoding="utf-8-sig"))
    result = {"ok": False, "python": sys.executable, "device": args.device,
              "runtime_version": manifest["version"], "loaded_modules": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        assert ctypes.sizeof(ctypes.c_void_p) == 8
        for entry in manifest["files"]:
            assert hashlib.sha256((root / entry["name"]).read_bytes()).hexdigest() == entry["sha256"], entry["name"]
        # Match the API's import order, since earlier native imports can influence
        # which DLL Windows reuses for a later dependency with the same basename.
        import numpy
        import soundfile
        import torch
        result["torch_version"] = torch.__version__
        if args.device == "cuda":
            assert torch.cuda.is_available()
            device = torch.device("cuda")
        elif args.device == "dml":
            import torch_directml
            device = torch_directml.device()
        else:
            device = torch.device("cpu")
        value = torch.tensor([[1., 2.], [3., 4.]], device=device)
        actual = (value @ value).cpu()
        assert torch.equal(actual, torch.tensor([[7., 10.], [15., 22.]]))
        for entry in manifest["files"]:
            name = entry["name"]
            loaded = loaded_dll_path(name)
            result["loaded_modules"][name] = loaded
            if loaded:
                assert Path(loaded).resolve() == root / name, f"Unexpected runtime DLL source: {name} -> {loaded}"
        for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            assert result["loaded_modules"][name], f"Core runtime was not loaded: {name}"
        result["ok"] = True
        print(f"PASS {root}: torch={torch.__version__}, backend={args.device}, all loaded VC DLLs are app-local")
    except Exception as exc:
        result["error"] = repr(exc)
        traceback.print_exc()
        return 1
    finally:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
