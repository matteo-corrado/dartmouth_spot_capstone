"""OpenCV/numpy re-implementation of BD's front stitch (no OpenGL).

Same geometry as stitch_front_images.py: virtual eye between the two front
cameras, plane 2 m ahead along the averaged camera axis, gluPerspective(110),
BD's blend score (pow 10). Runs in spot-env (cv2 + bosdyn already there).
"""
import io, pathlib, sys, time
import cv2
import numpy as np
from PIL import Image
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b

HERE = pathlib.Path(__file__).parent
W, H, FOVY, DIST = 1080, 720, 110.0, 2.0


def load(path):
    r = image_pb2.GetImageResponse()
    r.ParseFromString(pathlib.Path(path).read_bytes())
    return {i.source.name: i for i in r.image_responses}


def cam(resp):
    img = np.asarray(Image.open(io.BytesIO(resp.shot.image.data)).convert("L"))
    body_T_s = get_a_tform_b(resp.shot.transforms_snapshot, BODY_FRAME_NAME, resp.shot.frame_name_image_sensor)
    k = resp.source.pinhole.intrinsics
    return dict(img=img, body_T_s=body_T_s, s_T_body=body_T_s.inverse().to_matrix(),
                f=(k.focal_length.x, k.focal_length.y), c=(k.principal_point.x, k.principal_point.y))


def score(u, v, cols, rows):
    # BD frag shader calculateScore, on 0..1 image coords
    lx = 1 - np.abs(u / cols * 2 - 1)
    ly = 1 - np.abs(v / rows * 2 - 1)
    s = lx * ly * (1 - 0.001) + 0.001
    s = np.where((lx < 0) | (ly < 0), 0.0, s)
    return s ** 10


def build_maps(c1, c2):
    """Per-camera remap grids + weights. Depends only on calibration, so it can be cached."""
    eye = np.array([c1["body_T_s"].x + c2["body_T_s"].x, c1["body_T_s"].y + c2["body_T_s"].y,
                    c1["body_T_s"].z + c2["body_T_s"].z])  # BD sums positions (not average) - kept as-is
    n = np.array(c1["body_T_s"].rot.transform_point(0, 0, 1)) + np.array(c2["body_T_s"].rot.transform_point(0, 0, 1))
    eye[1] = 0; n[1] = 0
    n /= np.linalg.norm(n)
    plane = eye + n * DIST
    f = n
    s = np.cross(f, [0, 0, 1]); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    fy = 1 / np.tan(np.radians(FOVY) / 2)
    xs = (2 * (np.arange(W) + 0.5) / W - 1) * (W / H) / fy
    ys = (1 - 2 * (np.arange(H) + 0.5) / H) / fy
    X, Y = np.meshgrid(xs, ys)
    d = f[None, None] + X[..., None] * s + Y[..., None] * u        # ray dirs, body frame
    t = ((plane - eye) @ n) / (d @ n)
    P = eye + t[..., None] * d                                       # points on plane
    Ph = np.concatenate([P, np.ones(P.shape[:2] + (1,))], -1)
    out = []
    for c in (c1, c2):
        q = Ph @ c["s_T_body"].T
        z = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = (c["f"][0] * q[..., 0] / z + c["c"][0]).astype(np.float32)
            mv = (c["f"][1] * q[..., 1] / z + c["c"][1]).astype(np.float32)
        rows, cols = c["img"].shape
        w = np.where(z > 0, score(mu, mv, cols, rows), 0.0)
        out.append((mu, mv, w))
    return out


def stitch(c1, c2, maps):
    acc = np.zeros((H, W)); tot = np.zeros((H, W))
    for c, (mu, mv, w) in zip((c1, c2), maps):
        # GL texel centres sit at (i+0.5)/N, so pixel index = coord - 0.5; GL also
        # never samples black outside the texture, so replicate the edge.
        acc += cv2.remap(c["img"], mu - 0.5, mv - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * w
        tot += w
    out = np.zeros((H, W))
    np.divide(acc, tot, out=out, where=tot > 0)  # edge weights reach ~1e-30; don't clamp
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    fr = load(sys.argv[1] if len(sys.argv) > 1 else HERE / "frames.bin")
    c1, c2 = cam(fr["frontright_fisheye_image"]), cam(fr["frontleft_fisheye_image"])
    t = time.time(); maps = build_maps(c1, c2); tb = time.time() - t
    t = time.time(); img = stitch(c1, c2, maps); ts = time.time() - t
    print(f"build maps {tb*1000:.0f} ms (once), stitch {ts*1000:.1f} ms per frame")
    cv2.imwrite(str(HERE / "cv_stitch.png"), img)
    bd = cv2.imread(str(HERE / "bd_stitch.png"), cv2.IMREAD_GRAYSCALE)
    if bd is not None:
        diff = np.abs(bd.astype(int) - img.astype(int))
        print(f"vs BD: mean abs diff {diff.mean():.1f}, pixels >20 off {100*(diff>20).mean():.1f}%")
        cv2.imwrite(str(HERE / "side_by_side.png"), np.hstack([bd, img]))
    print("saved", HERE / "cv_stitch.png")
