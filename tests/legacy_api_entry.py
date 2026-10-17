"""Run a separately supplied baseline API/inference revision in memory.

The baseline directory must contain ``api_server.py`` and ``run_inference.py``.
The current checkout supplies the surrounding package, model files and runtime;
only the two baseline source files are loaded from the explicit directory.
"""
import argparse
import importlib.abc
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_wrapper_args() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Directory containing the baseline api_server.py and run_inference.py.",
    )
    if any(value in ("-h", "--help") for value in sys.argv[1:]):
        parser.print_help()
        raise SystemExit(0)
    args, remaining = parser.parse_known_args()
    baseline_root = args.baseline_root
    if baseline_root is None:
        raise SystemExit("--baseline-root is required for the legacy comparison entry.")
    baseline_root = baseline_root.resolve()
    missing = [name for name in ("api_server.py", "run_inference.py")
               if not (baseline_root / name).is_file()]
    if missing:
        raise SystemExit(
            f"Baseline root {baseline_root} is missing: {', '.join(missing)}"
        )
    # api_server.py owns the remaining command-line options. Do not let the
    # wrapper's option leak into its argparse parser.
    sys.argv = [sys.argv[0], *remaining]
    return baseline_root


BASELINE_ROOT = parse_wrapper_args()


def source(name):
    return (BASELINE_ROOT / f"{name}.py").read_text(encoding="utf-8-sig")


class LegacyInferenceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "run_inference":
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(ROOT / "run_inference.py")
        exec(compile(source("run_inference"), module.__file__, "exec"), module.__dict__)


sys.meta_path.insert(0, LegacyInferenceLoader())
__file__ = str(ROOT / "api_server.py")
exec(compile(source("api_server"), __file__, "exec"), globals())
