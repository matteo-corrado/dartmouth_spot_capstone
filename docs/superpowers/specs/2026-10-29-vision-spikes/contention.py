"""THROWAWAY: Gemma (llama-server) speed vs YOLO TRT load, and YOLO latency while Gemma generates."""
import threading, time, statistics, requests, sys
import bench

URL = "http://localhost:11435/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolo26l-seg"


def gemma():
    r = requests.post(URL, json={"messages": [{"role": "user", "content": "Write 120 words about robot dogs."}],
                                 "max_tokens": 150, "temperature": 0.0,
                                 "chat_template_kwargs": {"enable_thinking": False}}, timeout=120).json()
    t = r["timings"]
    return t["predicted_per_second"], t["prompt_per_second"]


def yolo_loop(stop, hz, lat):
    s = bench.ort.InferenceSession(str(bench.MODELS / f"{MODEL}.onnx"), providers=bench.providers("trt"))
    x = bench.prep(bench.IMG); n = s.get_inputs()[0].name
    while not stop.is_set():
        t = time.time(); s.run(None, {n: x}); dt = time.time() - t
        lat.append(dt * 1000)
        if hz:
            time.sleep(max(0, 1 / hz - dt))


def trial(hz):
    stop, lat = threading.Event(), []
    th = None
    if hz is not None:
        th = threading.Thread(target=yolo_loop, args=(stop, hz, lat), daemon=True); th.start(); time.sleep(6)
        lat.clear()
    res = [gemma() for _ in range(2)]
    stop.set()
    if th: th.join()
    tok = statistics.mean(r[0] for r in res)
    yl = f"yolo {statistics.median(lat):.1f} ms median, p90 {sorted(lat)[int(len(lat)*.9)]:.1f}" if lat else ""
    print(f"yolo rate {('off' if hz is None else ('max' if hz == 0 else f'{hz} img/s')):>10}: gemma {tok:5.1f} tok/s  {yl}", flush=True)


if __name__ == "__main__":
    for hz in (None, 10, 30, 50, 0):
        trial(hz)
