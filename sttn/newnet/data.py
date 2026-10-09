# Dataset adapter for FGFI-Net. Reuses sttn/train/synth.py (frame folders from `synth.py prep`, Chinese subtitle
# styles, renderer, phone-like stroke masks) and adds what the new net needs:
#   * arbitrary aligned crop sizes (no 432x240 lock), T consecutive frames (temporal stride 1-2)
#   * watermarks: static and moving (linear drift, like the moving "花木林" mark), semi-transparent, outlined
#   * 1-2 subtitle lines, line switches inside the window, JPEG on the corrupted input
#   * "prefill": part of the hole already holds slightly-off true background, as after the App's pixel propagation
#   * flows exactly like the App: DIS PRESET_FAST on the *corrupted* band at 1/2 res, hole filled by normalised
#     convolution (FlowField.java); ground-truth flows for the losses from DIS PRESET_MEDIUM on the clean frames
# Curriculum: `level` (0..1, shared with worker processes) scales augmentation difficulty.
import os, sys, glob, random, multiprocessing as mp
import cv2, numpy as np, torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train'))
import synth  # noqa: E402  (sttn/train/synth.py)

LATIN = 'abcdefghijklmnopqrstuvwxyz0123456789'


def find_fonts(extra=None):
    fs = list(extra or []) + synth.default_fonts()
    for pat in ['/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc', '/usr/share/fonts/opentype/noto/NotoSerifCJK-*.ttc',
                '/usr/share/fonts/truetype/wqy/*.ttc', '/usr/share/fonts/**/*CJK*.ttc', '/usr/share/fonts/**/*CJK*.otf']:
        fs += glob.glob(pat, recursive=True)
    fs = sorted(set(fs))
    assert fs, 'no CJK font found: pass --fonts or install fonts-noto-cjk'
    return fs


def _restyle(st, size, alpha=None):
    st.size = int(size)
    st.stroke = min(st.stroke, max(1, st.size // 12))
    st.font = synth.ImageFont.truetype(st.font_path, st.size)
    if alpha is not None:
        st.alpha = alpha
    return st


def wm_text(rng):
    cs = synth.common_hanzi()
    r = rng.random()
    if r < 0.4:
        return ''.join(rng.choice(cs) for _ in range(rng.randint(2, 5)))
    if r < 0.7:
        return '@' + ''.join(rng.choice(cs) for _ in range(rng.randint(2, 4)))
    if r < 0.85:
        return ''.join(rng.choice(cs) for _ in range(2)) + '号:' + ''.join(rng.choice(LATIN) for _ in range(rng.randint(5, 9)))
    return ''.join(rng.choice(LATIN) for _ in range(rng.randint(4, 10)))


def nc_fill(f, hole, sigma):
    """App FlowField: replace flow inside the (dilated) hole by Gaussian normalised convolution of the valid flow."""
    hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
    valid = (hole == 0).astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(valid, (0, 0), sigma), 1e-4)
    out = f.copy()
    for c in range(2):
        fill = cv2.GaussianBlur(f[..., c] * valid, (0, 0), sigma) / den
        out[..., c] = np.where(valid > 0, f[..., c], fill)
    return out


class _DIS:
    _d = {}

    @classmethod
    def get(cls, preset):
        if preset not in cls._d:
            cls._d[preset] = cv2.DISOpticalFlow_create(preset)
        return cls._d[preset]


def dis_pair(a, b, preset):
    d = _DIS.get(preset)
    return d.calc(a, b, None), d.calc(b, a, None)


class Clips(Dataset):
    """Returns dict of float tensors: gt/inp [T,3,H,W] 0..1, masks [T,2,H,W] (unknown, prefilled),
    ff/fb [T-1,2,H/2,W/2] phone-style flows (t->t+1 / t+1->t, half-res px), gff/gfb same from clean frames."""

    def __init__(self, root, fonts=None, T=8, crop=(192, 432), length=10 ** 7, seed=None, level=None, scale=(0.75, 1.25),
                 gt_flow=True):
        self.clips = [d for d in sorted(glob.glob(os.path.join(root, '*'))) if os.path.isdir(d)]
        self.frames = {d: sorted(glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.png'))) for d in self.clips}
        self.clips = [d for d in self.clips if len(self.frames[d]) >= 2 * T + 2]
        assert self.clips, 'no frame folders (synth.py prep) under ' + root
        self.fonts = find_fonts(fonts)
        self.T, (self.H, self.W), self.length, self.seed, self.scale = T, crop, length, seed, scale
        assert self.H % 12 == 0 and self.W % 72 == 0, 'crop must be H%12==0, W%72==0'
        self.level = level if level is not None else mp.Value('d', 1.0)
        self.gt_flow = gt_flow

    def __len__(self):
        return self.length

    # ---- overlays -----------------------------------------------------------------------------------------------
    def _subtitles(self, rng, L):
        H, W, T = self.H, self.W, self.T
        st = synth.SubtitleStyle(rng, self.fonts)
        _restyle(st, min(st.size, int(H * 0.38)))
        two = rng.random() < 0.1 + 0.3 * L and st.size * 2.6 < H
        cy = rng.uniform(st.size * 0.8, H - st.size * (2.0 if two else 0.8))
        cx = W / 2 + rng.uniform(-0.15, 0.15) * W
        lines = [[synth.random_line(rng)] + ([synth.random_line(rng)] if two else [])]
        switch = rng.randint(1, T - 1) if rng.random() < 0.3 else None
        if switch:
            lines.append([synth.random_line(rng)] + ([synth.random_line(rng)] if two else []))
        cache, ovs = {}, []
        for t in range(T):
            k = 1 if switch and t >= switch else 0
            if k not in cache:
                cache[k] = [synth.render_line(s, st, W, H, cx, cy + j * st.size * 1.3) for j, s in enumerate(lines[k])]
            ovs.append([(o, st.size) for o in cache[k]])
        return ovs

    def _watermark(self, rng, moving):
        H, W, T = self.H, self.W, self.T
        st = synth.SubtitleStyle(rng, self.fonts)
        st.box, st.shadow = False, (st.shadow if rng.random() < 0.5 else None)
        _restyle(st, rng.uniform(16, min(48, H * 0.3)), alpha=rng.uniform(0.3, 0.9))
        st.fill = (255, 255, 255) if rng.random() < 0.8 else tuple(rng.randint(120, 255) for _ in range(3))
        text = wm_text(rng)
        cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(st.size, H - st.size)
        vx, vy = (rng.uniform(-6, 6), rng.uniform(-2, 2)) if moving else (0, 0)
        ovs, still = [], None
        for t in range(T):
            if moving:
                ovs.append([(synth.render_line(text, st, W, H, cx + vx * t, cy + vy * t), st.size)])
            else:
                still = synth.render_line(text, st, W, H, cx, cy) if still is None else still
                ovs.append([(still, st.size)])
        return ovs

    # ---- sample ---------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        rng = random.Random(None if self.seed is None else self.seed * 1000003 + idx)
        nrs = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
        L = float(self.level.value)
        H, W, T = self.H, self.W, self.T
        d = rng.choice(self.clips)
        fs = self.frames[d]
        step = 2 if (rng.random() < 0.3 and len(fs) >= 2 * T + 2) else 1
        s0 = rng.randint(0, len(fs) - 1 - step * (T - 1))
        imgs = [cv2.imread(fs[s0 + step * t])[..., ::-1] for t in range(T)]
        h, w = imgs[0].shape[:2]
        s = rng.uniform(*self.scale)
        cw, ch = round(W * s), round(H * s)
        if cw > w or ch > h:                                  # small source: take the largest crop that fits
            f = min(w / cw, h / ch)
            cw, ch = int(cw * f), int(ch * f)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        flip = rng.random() < 0.5
        gt = []
        for im in imgs:
            c = im[y0:y0 + ch, x0:x0 + cw]
            if (cw, ch) != (W, H):
                c = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA if cw > W else cv2.INTER_CUBIC)
            gt.append(np.ascontiguousarray(c[:, ::-1] if flip else c).astype(np.float32) / 255)

        layers = [self._subtitles(rng, L)]
        if rng.random() < 0.15 + 0.45 * L:
            layers.append(self._watermark(rng, moving=rng.random() < 0.3 + 0.3 * L))
        sub_size = layers[0][0][0][1]
        inp, holes = [], []
        jpeg_q = rng.randint(55, 95) if rng.random() < 0.6 * L else None
        dil = rng.uniform(0.7, 1.6)
        for t in range(T):
            x = gt[t]
            m = np.zeros((H, W), np.uint8)
            for lay in layers:
                for ov, size in lay[t]:
                    x = synth.compose(x, ov)
                    mm = synth.overlay_mask(ov, max(8, size * dil))
                    if rng.random() < 0.05:                     # detector-box style hole
                        ys, xs = np.nonzero(mm)
                        if len(xs):
                            mm[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5] = 1
                    m |= mm
            x = (x * 255 + 0.5).clip(0, 255).astype(np.uint8)
            if jpeg_q:
                x = cv2.imdecode(cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])[1], cv2.IMREAD_COLOR)
            inp.append(x)
            holes.append(m)

        # phone-style flow on the corrupted band (before any hole handling), half resolution
        hs, ws_ = H // 2, W // 2
        gray = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (ws_, hs), interpolation=cv2.INTER_AREA) for x in inp]
        msm = [cv2.resize(m, (ws_, hs), interpolation=cv2.INTER_NEAREST) for m in holes]
        sigma = max(4.0, sub_size * 0.5 * 0.8)
        ff, fb, gff, gfb = [], [], [], []
        for t in range(T - 1):
            a, b = dis_pair(gray[t], gray[t + 1], cv2.DISOPTICAL_FLOW_PRESET_FAST)
            hole = msm[t] | msm[t + 1]
            ff.append(nc_fill(a, hole, sigma))
            fb.append(nc_fill(b, hole, sigma))
        if self.gt_flow:
            gg = [cv2.resize(cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (ws_, hs), interpolation=cv2.INTER_AREA)
                  for g in gt]
            for t in range(T - 1):
                a, b = dis_pair(gg[t], gg[t + 1], cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                gff.append(a)
                gfb.append(b)

        # prefill: part of the hole already carries slightly-off true background (App pixel propagation)
        unk, pre = [], []
        inpf = [x.astype(np.float32) / 255 for x in inp]
        do_pre = rng.random() < 0.5 * L
        for t in range(T):
            m = holes[t].astype(np.float32)
            p = np.zeros_like(m)
            if do_pre and m.any():
                noise = cv2.resize(nrs.rand(max(2, H // 24), max(2, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
                p = m * (noise > rng.uniform(0.3, 0.8))
                if p.any():
                    dx, dy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
                    src = cv2.warpAffine(gt[t], np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
                    src = np.clip(src * rng.uniform(0.95, 1.05), 0, 1)
                    if rng.random() < 0.3:
                        src = cv2.GaussianBlur(src, (0, 0), rng.uniform(0.5, 1.2))
                    inpf[t] = inpf[t] * (1 - p[..., None]) + src * p[..., None]
            u = m * (1 - p)
            inpf[t] = inpf[t] * (1 - u[..., None])
            unk.append(u)
            pre.append(p)

        tt = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a))).permute(0, 3, 1, 2).float()
        out = {'gt': tt(gt), 'inp': tt(inpf), 'masks': torch.from_numpy(np.stack([np.stack(unk), np.stack(pre)], 1)).float(),
               'ff': tt(ff), 'fb': tt(fb)}
        if self.gt_flow:
            out['gff'], out['gfb'] = tt(gff), tt(gfb)
        return out


def worker_init(_):
    cv2.setNumThreads(1)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='write a contact sheet of training samples')
    ap.add_argument('root')
    ap.add_argument('--crop', default='192x432')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--level', type=float, default=1.0)
    ap.add_argument('--out', default='newnet_preview.jpg')
    a = ap.parse_args()
    H, W = map(int, a.crop.split('x'))
    ds = Clips(a.root, crop=(H, W), seed=3, level=mp.Value('d', a.level))
    rows = []
    for i in range(a.n):
        s = ds[i]
        k = ds.T // 2
        u8 = lambda z: (z.permute(1, 2, 0).numpy()[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
        vis = u8(s['inp'][k]).copy()
        vis[s['masks'][k, 0].numpy() > 0] = (vis[s['masks'][k, 0].numpy() > 0] * 0.5 + np.array([0, 0, 127])).astype(np.uint8)
        vis[s['masks'][k, 1].numpy() > 0] = (vis[s['masks'][k, 1].numpy() > 0] * 0.5 + np.array([0, 127, 0])).astype(np.uint8)
        fl = s['ff'][k].permute(1, 2, 0).numpy()
        hsv = np.zeros(fl.shape[:2] + (3,), np.uint8)
        mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        hsv[..., 0], hsv[..., 1], hsv[..., 2] = ang * 90 / np.pi, 255, np.clip(mag * 20, 0, 255)
        flv = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (W, H))
        rows.append(np.hstack([u8(s['gt'][k]), vis, flv]))
    cv2.imwrite(a.out, np.vstack(rows))
    print('wrote', a.out, '(gt | input: red=unknown hole, green=prefilled | phone flow t->t+1)')
