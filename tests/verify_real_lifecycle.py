"""Opt-in real-weight regression: baseline/current parity and device moves.

The baseline source is supplied explicitly with ``--baseline-root`` and is
loaded in memory only. The active voice configuration determines all model and
reference paths, so the check follows the same paths as the service.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path,
                        help="Optional baseline run_inference.py directory; omitted uses current engine.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gpt-device", choices=["cpu", "cuda"])
    parser.add_argument("--avatar", default="ami")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_root = args.baseline_root.resolve() if args.baseline_root else None
    if baseline_root and not (baseline_root / "run_inference.py").is_file():
        parser.error(f"baseline root is missing run_inference.py: {baseline_root}")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("output already exists; use a new report path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('{"ok": false, "status": "running"}', encoding="utf-8")
    import numpy as np
    import torch
    # Half-precision cuDNN can select nondeterministic convolution algorithms;
    # use deterministic kernels for a strict numerical regression, not timing.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from run_inference import GPTSoVITSInference

    voices = json.loads((ROOT / "resources/voices.json").read_text(encoding="utf-8"))
    if args.avatar not in voices:
        parser.error(f"unknown avatar: {args.avatar}")
    voice = voices[args.avatar]
    options = dict(gpt_path=str(ROOT / "resources" / voice["gpt_path"]),
                   sovits_path=str(ROOT / "resources" / voice["sovits_path"]),
                   prompt_cache_path=str(ROOT / "resources" / voice["references"]["default"]["cache_path"]),
                   device=args.device, gpt_device=args.gpt_device, cpu_threads=4)
    baseline_class = GPTSoVITSInference
    if baseline_root:
        source = (baseline_root / "run_inference.py").read_bytes()
        baseline = types.ModuleType("tts_baseline_regression")
        baseline.__file__ = str(ROOT / "run_inference.py")
        sys.modules[baseline.__name__] = baseline
        exec(compile(source.decode("utf-8-sig"), baseline.__file__, "exec"), baseline.__dict__)
        baseline_class = baseline.GPTSoVITSInference
    result = {"device": args.device, "gpt_device": args.gpt_device, "avatar": args.avatar,
              "baseline_root": str(baseline_root) if baseline_root else None, "cycles": []}

    def synthesize(engine):
        torch.manual_seed(20260909)
        np.random.seed(20260909)
        return engine.synthesize("こんにちは、今日もよろしくお願いします。", return_timeline=True,
                                 top_k=15, top_p=1, temperature=1, pause_length=0)

    def capture_semantics(engine):
        original = engine._generate_segment_semantic
        def capture(*positional, **keywords):
            value = original(*positional, **keywords)
            engine.test_semantic = value["pred_semantic"].detach().cpu().clone()
            return value
        engine._generate_segment_semantic = capture

    torch.manual_seed(1234)
    old = baseline_class(**options)
    capture_semantics(old)
    # Prime the exact acoustic shape before comparing steady-state numerics.
    # The pre-change engine itself can differ on its first shape-specific run.
    synthesize(old)
    reference_audio, reference_timestamps = synthesize(old)
    reference_semantic = old.test_semantic
    repeated_audio, _ = synthesize(old)
    print("Baseline self repeat max difference", float(np.max(np.abs(reference_audio-repeated_audio))), flush=True)
    expected_weights = {name: tensor.detach().cpu().clone() for name, tensor in old.t2s_model.state_dict().items()}
    expected_acoustic = {name: tensor.detach().cpu().clone() for name, tensor in old.vq_model.state_dict().items()}
    old.close()
    del old

    torch.manual_seed(1234)
    engine = GPTSoVITSInference(**options)
    capture_semantics(engine)
    actual = engine.t2s_model.state_dict()
    assert set(actual) == set(expected_weights), "Checkpoint keys changed"
    for name in actual:
        assert torch.equal(actual[name].cpu(), expected_weights[name]), name
    for name, tensor in engine.vq_model.state_dict().items():
        assert torch.equal(tensor.cpu(), expected_acoustic[name]), name
    del expected_acoustic
    del expected_weights, actual
    synthesize(engine)
    current_audio, current_timestamps = synthesize(engine)
    assert torch.equal(reference_semantic, engine.test_semantic), "Semantic tokens changed"
    result["legacy_audio_max_abs_difference"] = float(np.max(np.abs(reference_audio-current_audio))) if reference_audio.shape == current_audio.shape else None
    result["legacy_audio_same_shape"] = reference_audio.shape == current_audio.shape
    result["legacy_timestamps_equal"] = reference_timestamps == current_timestamps
    np.testing.assert_allclose(current_audio, reference_audio, rtol=1e-4, atol=1e-5)
    assert current_timestamps == reference_timestamps, "Legacy timestamps changed"
    print("Legacy/current parity passed", flush=True)

    for index in range(args.cycles):
        started = time.perf_counter()
        engine.hibernate()
        engine.hibernate()
        sleep_seconds = time.perf_counter()-started
        assert engine.is_hibernated
        for model in (engine.t2s_model, engine.vq_model):
            assert all(t.device.type == "cpu" for t in list(model.parameters()) + list(model.buffers()))
        for name in ("prompt_semantic", "refer_spec", "sv_emb", "bert1"):
            assert getattr(engine.prompt_cache, name).device.type == "cpu"
        sleeping_vram = torch.cuda.memory_allocated() / 2**20 if args.device == "cuda" else None
        started = time.perf_counter()
        engine.wake()
        engine.wake()
        wake_seconds = time.perf_counter()-started
        audio, timestamps = synthesize(engine)
        np.testing.assert_allclose(audio, current_audio, rtol=1e-4, atol=1e-5)
        assert timestamps == current_timestamps
        result["cycles"].append({"sleep_seconds": sleep_seconds, "wake_seconds": wake_seconds,
                                 "sleeping_cuda_allocated_mb": sleeping_vram,
                                 "audio_max_abs_difference": float(np.max(np.abs(audio-current_audio)))})
        print(f"Parity cycle {index+1}/{args.cycles} passed", flush=True)
    engine.close()
    result["ok"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
