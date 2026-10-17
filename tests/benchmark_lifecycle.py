"""Isolated real-model HTTP benchmark; it starts its own server and touches no other process.

Run with an embedded Python. Results include process startup, warm synthesis,
sleep/wake and Windows process memory. Use --baseline before implementing sleep.
The optional legacy arm loads source files from an explicit baseline directory.
"""
import argparse
import base64
import ctypes
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def process_memory(pid):
    if os.name != "nt":
        return {}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return {"error": ctypes.get_last_error()}
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {"error": ctypes.get_last_error()}
        return {"working_set_mb": counters.WorkingSetSize / 2**20,
                "private_commit_mb": counters.PrivateUsage / 2**20,
                "page_faults": counters.PageFaultCount}
    finally:
        kernel.CloseHandle(handle)


def trim_working_set(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x0400 | 0x0100, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not psapi.EmptyWorkingSet(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpt-device")
    parser.add_argument("--avatar", default="ami")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--legacy-tracked-code", action="store_true")
    parser.add_argument("--baseline-root", type=Path,
                        help="Directory containing baseline api_server.py and run_inference.py.")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--idle-seconds", type=float, default=1)
    parser.add_argument("--trim-working-set", action="store_true")
    parser.add_argument("--exercise-api", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-root", type=Path, default=ROOT)
    args = parser.parse_args()
    server_root = args.server_root.resolve()
    if args.legacy_tracked_code and args.baseline_root is None:
        parser.error("--legacy-tracked-code requires --baseline-root")
    if args.baseline_root is not None:
        args.baseline_root = args.baseline_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('{"ok": false, "status": "running"}', encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_hex(16)
    entry = ROOT / "tests/legacy_api_entry.py" if args.legacy_tracked_code else server_root / "api_server.py"
    command = [sys.executable, "-u", str(entry)]
    if args.legacy_tracked_code:
        command += ["--baseline-root", str(args.baseline_root)]
    command += ["--port", str(port),
               "--token", token, "--avatar", args.avatar, "--reference", "default",
               "--device", args.device, "--cpu_threads", "4", "--resources", str(server_root / "resources")]
    if args.gpt_device:
        command += ["--gpt_device", args.gpt_device]
    result = {"device": args.device, "gpt_device": args.gpt_device, "avatar": args.avatar,
              "python": sys.executable, "baseline": args.baseline, "trim_working_set": args.trim_working_set,
              "idle_seconds": args.idle_seconds, "cycles": [], "synthesis": []}
    result["legacy_tracked_code"] = args.legacy_tracked_code
    result["baseline_root"] = str(args.baseline_root) if args.baseline_root else None
    result["server_root"] = str(server_root)
    log = args.output.with_suffix(".log").open("w", encoding="utf-8")
    lines = queue.Queue()
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=server_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONUTF8": "1"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    def read_output():
        for line in process.stdout:
            log.write(f"{time.perf_counter()-started:.3f} {line}")
            log.flush()
            if line.startswith("GSV_TTS_STARTUP "):
                lines.put(line)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def request(path, payload=None):
        before = time.perf_counter()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
        data = json.loads(body) if "json" in content_type else {"bytes": len(body)}
        if isinstance(data, dict) and "audio" in data:
            wav = base64.b64decode(data.pop("audio"))
            if wav[:4] != b"RIFF" or len(wav) <= 44:
                raise AssertionError("Missing valid WAV audio")
            data["audio_bytes"] = len(wav)
            data["timestamp_count"] = len(data.pop("timestamps", []))
        return {"seconds": time.perf_counter()-before, "response": data}

    try:
        deadline = started + 600
        while time.perf_counter() < deadline:
            try:
                line = lines.get(timeout=1)
                ready = json.loads(line[len("GSV_TTS_STARTUP "):])
                if not ready["ok"]:
                    raise RuntimeError(str(ready))
                result["startup_seconds"] = time.perf_counter()-started
                break
            except queue.Empty:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {process.returncode}")
        else:
            raise TimeoutError("Startup exceeded 600 seconds")
        print("startup", result["startup_seconds"], flush=True)
        result["warmup_request"] = request("/api/tts/predict", {"text": "こんにちは。"})
        for _ in range(3):
            result["synthesis"].append(request("/api/tts/predict-timeline", {"text": "こんにちは、今日もよろしくお願いします。"}))
        result["hot_memory"] = process_memory(process.pid)
        result["hot_status"] = request("/api/tts/status")
        if not args.baseline:
            for index in range(args.cycles):
                cycle = {"hibernate": request("/api/tts/hibernate", {})}
                cycle["asleep_memory"] = process_memory(process.pid)
                cycle["asleep_status"] = request("/api/tts/status")
                if args.trim_working_set:
                    trim_started = time.perf_counter()
                    trim_working_set(process.pid)
                    cycle["trim_seconds"] = time.perf_counter() - trim_started
                    cycle["trimmed_memory"] = process_memory(process.pid)
                time.sleep(args.idle_seconds)
                cycle["wake"] = request("/api/tts/wake", {})
                cycle["synthesis"] = request("/api/tts/predict-timeline", {"text": "こんにちは、今日もよろしくお願いします。"})
                cycle["hot_memory"] = process_memory(process.pid)
                result["cycles"].append(cycle)
                print("cycle", index, "wake", cycle["wake"]["seconds"], "synthesis", cycle["synthesis"]["seconds"], flush=True)
        if args.exercise_api:
            checks = []
            for path in ("/api/tts/hibernate", "/api/tts/wake"):
                unauthorized = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"{}",
                                                      headers={"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(unauthorized, timeout=30)
                    raise AssertionError("Lifecycle endpoint accepted missing token")
                except urllib.error.HTTPError as exc:
                    assert exc.code == 401, exc.code
            checks.append("lifecycle_requires_authentication")
            for _ in range(2):
                assert request("/api/tts/hibernate", {})["response"]["hibernated"]
            for _ in range(3):
                assert request("/api/tts/status")["response"]["runtime_state"] == "hibernated"
            checks.append("hibernate_idempotent_and_status_does_not_wake")
            implicit = request("/api/tts/predict-timeline", {"text": "こんにちは。"})
            assert implicit["response"]["audio_bytes"] > 44
            assert request("/api/tts/status")["response"]["ready"]
            checks.append("synthesis_implicitly_wakes")
            other = "tyc" if args.avatar == "ami" else "ami"
            request("/api/tts/hibernate", {})
            assert request("/api/tts/wake", {"modelId": other, "referenceId": "default"})["response"]["active_model_id"] == other
            assert request("/api/tts/predict-timeline", {"text": "こんにちは。"})["response"]["active_model_id"] == other
            request("/api/tts/hibernate", {})
            assert request("/api/tts/wake", {"modelId": args.avatar, "referenceId": "default"})["response"]["active_model_id"] == args.avatar
            checks.append("switch_voice_while_asleep_and_switch_back")
            with ThreadPoolExecutor(max_workers=2) as executor:
                synthesis = executor.submit(request, "/api/tts/predict-timeline", {"text": "こんにちは、今日もよろしくお願いします。"})
                sleeping = executor.submit(request, "/api/tts/hibernate", {})
                assert synthesis.result()["response"]["audio_bytes"] > 44
                assert sleeping.result()["response"]["hibernated"]
            checks.append("concurrent_synthesis_and_hibernate")
            # The lock must remain held while Starlette consumes the stream.
            stream_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/tts/predict-stream",
                data=json.dumps({"text": "こんにちは、今日もよろしくお願いします。", "chunkLength": 24}).encode(),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            with urllib.request.urlopen(stream_req, timeout=300) as stream:
                assert json.loads(stream.readline())["type"] == "start"
                with ThreadPoolExecutor(max_workers=1) as executor:
                    sleeping = executor.submit(request, "/api/tts/hibernate", {})
                    events = [json.loads(line) for line in stream if line.strip()]
                    assert any(event["type"] == "done" for event in events)
                    assert not any(event["type"] == "error" for event in events)
                    assert sleeping.result()["response"]["hibernated"]
            assert request("/api/tts/status")["response"]["hibernated"]
            checks.append("stream_finishes_before_hibernation")
            result["api_checks"] = checks
            print("API regression checks", checks, flush=True)
        result["ok"] = True
    except Exception as exc:
        result["ok"] = False
        result["error"] = repr(exc)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=15)
        reader.join(timeout=5)
        log.close()
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
