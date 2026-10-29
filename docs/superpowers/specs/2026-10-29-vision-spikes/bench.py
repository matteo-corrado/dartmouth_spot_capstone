"""THROWAWAY: YOLO26 ONNX latency on Orin via onnxruntime-gpu (TRT fp16 / CUDA / CPU).

Run in spot-env:  spot-env/bin/python bench.py [model ...] [--cpu]
"""
import pathlib, sys, time
import numpy as np, cv2
import onnxruntime as ort

HERE = pathlib.Path(__file__).parent
MODELS = HERE / "models"
CACHE = HERE / "trt_cache"; CACHE.mkdir(exist_ok=True)
IMG = HERE.parent / "lookaround" / "hand_color_image.jpg"  # colour frame with a person


def prep(path):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]; r = 640 / max(h, w)
    im = cv2.resize(im, (int(w * r), int(h * r)))
    pad = np.full((640, 640, 3), 114, np.uint8); pad[:im.shape[0], :im.shape[1]] = im
    return np.ascontiguousarray(pad[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255)


def providers(kind):
    if kind == "trt":
        return [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                               "trt_engine_cache_path": str(CACHE), "trt_timing_cache_enable": True}),
                "CUDAExecutionProvider"]
    if kind == "cuda":
        return ["CUDAExecutionProvider"]
    return ["CPUExecutionProvider"]


def run(model, kind, x, n):
    t = time.time()
    s = ort.InferenceSession(str(MODELS / f"{model}.onnx"), providers=providers(kind))
    load = time.time() - t
    name = s.get_inputs()[0].name
    t = time.time()
    while time.time() - t < (1 if kind == "cpu" else 3):  # let DVFS clocks ramp up
        out = s.run(None, {name: x})
    t = time.time()
    for _ in range(n):
        out = s.run(None, {name: x})
    ms = (time.time() - t) / n * 1000
    det = out[0][0]
    keep = det[det[:, 4] > 0.4]
    return load, ms, keep, s.get_providers()[0]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    models = args or ["yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26l"]
    kinds = ["trt", "cuda"] + (["cpu"] if "--cpu" in sys.argv else [])
    x = prep(IMG)
    for m in models:
        ref = None
        for k in kinds:
            load, ms, keep, prov = run(m, k, x, 5 if k == "cpu" else 50)
            classes = sorted(set(int(c) for c in keep[:, 5]))
            agree = ""
            if ref is not None and len(ref) and len(keep):
                agree = f" box diff vs {kinds[0]} {np.abs(ref[0, :4] - keep[0, :4]).max():.1f}px"
            ref = keep if ref is None else ref
            print(f"{m:14s} {k:4s} ({prov[:4]}) {ms:7.1f} ms/img  load {load:5.1f}s  dets {len(keep)} classes {classes}{agree}", flush=True)
