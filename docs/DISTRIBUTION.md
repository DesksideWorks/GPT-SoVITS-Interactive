# Source checkout and Windows package

The source repository contains the engine, HTTP API, tools, test code, configuration,
three README translations and license notices. The complete Windows package also
includes a Python environment, voice weights, reference features, semantic conditions,
language models, dictionaries and Microsoft runtime libraries. It is released in a CUDA
edition (`Runtime/Python_CUDA/`) and a DirectML edition (`Runtime/Python_DirectML/` and
the GPT's `.ort-dml/` graphs); everything else is the same in both.

For a runnable local source checkout, copy the following paths from a matching
complete package while preserving their relative locations:

- `Runtime/`
- `resources/models/`, `resources/nltk_data/`
- `resources/avatars/ami/references/`, `resources/avatars/tyc/references/`
- `resources/semantic_prompts/*.npz`
- `GPT_SoVITS/text/ja_userdic/`, `GPT_SoVITS/text/*.pickle`
- `Redistributables/VC_Runtime_x64/*.dll`

These paths are excluded from Git. Keep the package notices and applicable model,
voice-source and dictionary licenses with the corresponding resources. Follow the
README for service startup. The requirements files are incremental dependencies,
not a lockfile for rebuilding the complete environment from scratch.

## Tools and tests

`tools/export_gpt_dml.py`, `tools/build_prompt_cache.py` and `tools/build_semantic_prompt.py`
prepare model-specific assets. ONNX graph export requires the separate preparation environment described
in the README; the embedded inference environments do not include the `onnx`
export package. The `test_*.py` files are unit tests; most `verify_*.py` scripts require real
weights and a suitable device. Write test reports/audio outside the package.

The lifecycle regression (`tests/verify_real_lifecycle.py`) uses the configured
voice paths and compares deterministic audio before and after repeated sleep/wake
cycles. It supports CPU/CUDA PyTorch engines; use `verify_semantic_prompt.py` or
`verify_chinese_api.py --device dml` for DirectML lifecycle coverage.
An optional `--baseline-root` selects a trusted directory containing an earlier
`run_inference.py`. This compares that file against the current engine with the
current dependencies and assets; it is not a comparison of two complete releases.

`tests/legacy_api_entry.py` loads both `api_server.py` and `run_inference.py` from
an explicit `--baseline-root`. The `--legacy-tracked-code --baseline-root ...`
options of `tests/benchmark_lifecycle.py` select this isolated comparison arm.
The option name is retained for compatibility; Git history is not read. For a
whole checkout comparison, use the benchmark's `--server-root` instead. Baseline
code executes as Python and must be trusted and compatible with current assets.

Examples from a complete package (use a new output path for each run):

```powershell
& ./Runtime/Python_CUDA/python.exe tests/verify_real_lifecycle.py --device cuda --cycles 2 --output ../checks/lifecycle.json
& ./Runtime/Python_DirectML/python.exe tests/verify_streaming_api.py --device dml --output ../checks/dml-stream
& ./Runtime/Python_CUDA/python.exe tests/verify_frontend_preparation.py --parity --output ../checks/bert.json
& ./Runtime/Python_DirectML/python.exe tools/verify_chinese_payload.py
```

`tools/sync_runtime_payload.py` updates an existing installation from an explicit
source directory and JSON file list. It has no Unity or game-directory dependency.
Stop the target service first. Use disjoint source, target and new backup directories,
and place the new report outside them. Preview with `--dry-run`, then repeat without
that flag to apply. Every replaced file is backed up and copied files are hashed.
Unlisted files remain untouched; runtime wheel replacements should use a fresh full
package to avoid leaving stale package files.

```json
{"files": ["api_server.py", "run_inference.py", "chinese-runtime-manifest.json"]}
```

```powershell
python tools/sync_runtime_payload.py --source ../package --target ../installed-tts --manifest ../update-files.json --backup ../update-backup --report ../update-report.json --dry-run
```

The report records original and new hashes and every attempted destination. To
undo an update, stop the service, restore replaced files from the backup, and
remove only additions recorded with a null `before_sha256`. Check that destination
hashes still match the update before reverting so later user changes are preserved.

The one-time upstream restoration script and private pre-prefix Chinese baseline
script are not distributed. Current Chinese behavior is covered by the semantic
prompt, multilingual frontend, BERT parity and HTTP checks above.

`tools/refresh_python_notices.py` regenerates the Python notice inventory from the
two embedded runtimes; `--check` verifies it without writing. Supplementary license
texts are retained under `Licenses/python/`.
