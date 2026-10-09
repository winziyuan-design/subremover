# Run FGFI-Net on real frames the way the App will: split the hole union of a T-frame window into regions, cut an aligned
# band around each (H % 12, W % 72; vertical marks rotated to horizontal), compute phone-style DIS flows on the band,
# run the net (torch checkpoint or exported ONNX) and paste the hole back.
import os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load, PhoneWrapper, align  # noqa: E402
import data  # noqa: E402


class Runner:
    def __init__(self, path, threads=4, device='cpu'):
        self.onnx = path.endswith('.onnx')
        if self.onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            self.s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
            shp = self.s.get_inputs()[0].shape
            self.fixed = (int(shp[2]), int(shp[3]))
            self.ah, self.aw = 12, 72
        else:
            g = load(path).eval().to(device)
            self.w = PhoneWrapper(g).eval()
            self.ah, self.aw = align(g.cfg)
            self.fixed = None
        self.device = device
        torch.set_num_threads(threads)

    def __call__(self, fr, m, ff, fb):
        if self.onnx:
            return self.s.run(None, {'frames': fr, 'masks': m, 'flow_fw': ff, 'flow_bw': fb})[0]
        with torch.no_grad():
            t = lambda z: torch.from_numpy(z).to(self.device)
            return self.w(t(fr), t(m), t(ff), t(fb)).cpu().numpy()


def phone_flows(band_rgb, holes, line_h):
    H, W = band_rgb[0].shape[:2]
    hs, ws = H // 2, W // 2
    gray = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (ws, hs), interpolation=cv2.INTER_AREA) for x in band_rgb]
    msm = [cv2.resize(m, (ws, hs), interpolation=cv2.INTER_NEAREST) for m in holes]
    sigma = max(4.0, line_h * 0.5 * 0.8)
    ff, fb = [], []
    for t in range(len(band_rgb) - 1):
        a, b = data.dis_pair(gray[t], gray[t + 1], cv2.DISOPTICAL_FLOW_PRESET_FAST)
        h = msm[t] | msm[t + 1]
        ff.append(data.nc_fill(a, h, sigma))
        fb.append(data.nc_fill(b, h, sigma))
    st = lambda a: np.ascontiguousarray(np.stack(a).transpose(0, 3, 1, 2)).astype(np.float32)
    return st(ff), st(fb)


def regions(masks, gap=24):
    """Boxes (x0,y0,x1,y1) of hole regions of the window (union over frames, merged when closer than `gap`)."""
    U = np.zeros_like(masks[0])
    for m in masks:
        U |= (m > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(U, np.ones((gap, gap), np.uint8)))
    return [(st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]) for i in range(1, n)]


def _axis(a0, a1, mult, full, ctx):
    """Aligned [s, s+len) covering [a0-ctx, a1+ctx) inside [0, full); several windows if longer than allowed."""
    need = (a1 - a0) + 2 * ctx
    L = int(np.ceil(need / mult)) * mult
    cap = (full // mult) * mult
    if L <= cap:
        c = (a0 + a1) // 2
        return [int(np.clip(c - L // 2, 0, full - L))], L
    n = int(np.ceil((need - mult) / (cap - mult)))      # overlapping windows of size cap
    lo, hi = max(0, a0 - ctx), min(full, a1 + ctx) - cap
    return [int(round(lo + i * (hi - lo) / max(1, n - 1))) for i in range(max(2, n))], cap


def run_window(runner, frames, masks, line_h=48, ctx=24, fixed=None):
    """frames: list of T full-res RGB uint8; masks: list of T uint8 (>0 = hole). Returns T output frames."""
    out = [f.copy() for f in frames]
    Hf, Wf = masks[0].shape
    for (x0, y0, x1, y1) in regions(masks):
        rot = (y1 - y0) > 1.5 * (x1 - x0)
        F_ = [np.ascontiguousarray(np.rot90(f)) for f in out] if rot else out
        M_ = [np.ascontiguousarray(np.rot90(m)) for m in masks] if rot else masks
        if rot:
            x0, y0, x1, y1 = y0, Wf - x1, y1, Wf - x0
        Hb, Wb = M_[0].shape
        if fixed:
            ys_, bh = [int(np.clip((y0 + y1) // 2 - fixed[0] // 2, 0, Hb - fixed[0]))], fixed[0]
            xs_, bw = _axis(x0, x1, fixed[1], Wb, ctx) if fixed[1] <= Wb else ([0], fixed[1])
        else:
            ys_, bh = _axis(y0, y1, runner.ah, Hb, ctx)
            xs_, bw = _axis(x0, x1, runner.aw, Wb, ctx)
        for by in ys_:
            for bx in xs_:
                band = [f[by:by + bh, bx:bx + bw] for f in F_]
                hole = [(m[by:by + bh, bx:bx + bw] > 0).astype(np.uint8) for m in M_]
                if not any(h.any() for h in hole):
                    continue
                ff, fb = phone_flows(band, hole, line_h)
                fr = np.stack(band).transpose(0, 3, 1, 2).astype(np.float32) / 255
                mk = np.zeros((len(band), 2, bh, bw), np.float32)
                mk[:, 0] = np.stack(hole)
                o = runner(fr, mk, ff, fb)
                o = (np.clip(o.transpose(0, 2, 3, 1), 0, 1) * 255 + 0.5).astype(np.uint8)
                for t in range(len(band)):
                    h3 = hole[t][..., None].astype(bool)
                    F_[t][by:by + bh, bx:bx + bw] = np.where(h3, o[t], band[t])
        if rot:
            out = [np.ascontiguousarray(np.rot90(f, -1)) for f in F_]
    return out
