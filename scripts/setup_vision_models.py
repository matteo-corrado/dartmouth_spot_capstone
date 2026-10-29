#!/usr/bin/env python3
"""Export the vision ONNX models off-env and warm their TensorRT engines.

Export runs in a separate venv (default /mnt/ssd/vision-export-env) because
ultralytics auto-installs packages and must never touch spot-env. Run this
script itself with spot-env's python; it shells out to the export venv.

Every model is exported with end2end=True (NMS-free head); runtime.decode
depends on that output layout. Idempotent: existing ONNX files are skipped.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VOCAB = REPO / "src" / "voice_control" / "vision" / "vocab.py"
MODELS = ("yolo26l-seg", "yolo26l-pose", "yoloe-26l-seg")
PACKAGES = ("ultralytics", "onnx", "onnxslim", "onnxruntime")

# Runs inside the export venv. argv: model name, output dir, vocab.py path.
_EXPORT_SNIPPET = r"""
import runpy, shutil, sys
from pathlib import Path
name, out, vocab = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
kw = dict(format="onnx", imgsz=640, opset=17, simplify=True, end2end=True)
if name.startswith("yoloe"):
    from ultralytics import YOLOE
    model = YOLOE(name + ".pt")
    model.set_classes(runpy.run_path(vocab)["YOLOE_NAMES"])
else:
    from ultralytics import YOLO
    model = YOLO(name + ".pt")
path = Path(model.export(**kw))
out.mkdir(parents=True, exist_ok=True)
shutil.move(str(path), str(out / (name + ".onnx")))
"""


def ensure_env(env: Path) -> Path:
    py = env / "bin" / "python"
    if not py.exists():
        print(f"creating export venv {env}")
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", *PACKAGES], check=True)
    return py


def export(env_py: Path, name: str, out: Path, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(env_py), "-c", _EXPORT_SNIPPET, name, str(out), str(VOCAB)],
                   cwd=work, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", type=Path, default=Path("/mnt/ssd/vision-export-env"))
    ap.add_argument("--out", type=Path, default=REPO / "models" / "vision")
    ap.add_argument("--no-warm", action="store_true", help="skip TensorRT engine warm-build")
    args = ap.parse_args(argv)

    t0 = time.time()
    out = args.out.resolve()
    missing = [m for m in MODELS if not (out / f"{m}.onnx").exists()]
    for m in MODELS:
        if m not in missing:
            print(f"skip export {m} (exists)")
    if missing:
        env_py = ensure_env(args.env)
        work = args.env / "work"
        for m in missing:
            print(f"exporting {m} ...")
            export(env_py, m, out, work)
        shutil.rmtree(work, ignore_errors=True)

    if args.no_warm:
        print("skip TensorRT warm-build (--no-warm)")
    else:
        sys.path.insert(0, str(REPO))
        from src.voice_control.vision.runtime import load_model
        for m in MODELS:
            t = time.time()
            model = load_model(m, model_dir=out)
            print(f"warm {m}: provider={model.provider} ({time.time() - t:.1f}s)")
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
