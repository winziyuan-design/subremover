# Synthetic subtitle data: random Chinese subtitle lines (outline / shadow / colour / size like phone videos) are drawn
# over clean footage, so the clean footage is the exact ground truth. Crops are taken at native resolution (scale ~1)
# because on the phone STTN runs on native-resolution 432x240 tiles of the subtitle band.
import os, glob, random, argparse
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from net import W_IN, H_IN

_GB = None


def common_hanzi():
    """GB2312 level-1 characters (3755 most common)."""
    global _GB
    if _GB is None:
        cs = []
        for hi in range(0xB0, 0xD8):
            for lo in range(0xA1, 0xFF):
                try:
                    cs.append(bytes([hi, lo]).decode('gb2312'))
                except UnicodeDecodeError:
                    pass
        _GB = cs
    return _GB


PUNCT = list('，。！？、：…') + [' ']


def random_line(rng):
    cs = common_hanzi()
    n = rng.randint(2, 16)
    s = ''.join(rng.choice(cs) for _ in range(n))
    if n > 5 and rng.random() < 0.4:
        k = rng.randint(2, n - 2)
        s = s[:k] + rng.choice(PUNCT) + s[k:]
    return s


class SubtitleStyle:
    def __init__(self, rng, fonts):
        self.font_path = rng.choice(fonts)
        self.size = int(rng.choice([rng.uniform(26, 44), rng.uniform(40, 64), rng.uniform(56, 80)]))
        r = rng.random()
        self.fill = (255, 255, 255) if r < 0.7 else (255, 230, 60) if r < 0.85 else tuple(rng.randint(150, 255) for _ in range(3))
        self.stroke = rng.choice([0, 0, 1, 2, 2, 3, 4]) * max(1, self.size // 40)
        self.stroke_fill = (0, 0, 0) if rng.random() < 0.85 else (40, 40, 40)
        self.shadow = (rng.randint(1, 4), rng.randint(1, 4)) if rng.random() < 0.4 else None
        self.box = rng.random() < 0.08                 # translucent background strip
        self.alpha = 1.0 if rng.random() < 0.85 else rng.uniform(0.6, 0.95)
        self.font = ImageFont.truetype(self.font_path, self.size)


def render_line(text, st, w, h, cx, cy):
    """RGBA overlay (h,w,4) float 0..1 with the line centred at (cx,cy)."""
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), text, font=st.font, stroke_width=st.stroke)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = int(cx - tw / 2 - bb[0]), int(cy - th / 2 - bb[1])
    if st.box:
        pad = st.size // 4
        d.rectangle([x + bb[0] - pad, y + bb[1] - pad, x + bb[2] + pad, y + bb[3] + pad], fill=(0, 0, 0, 110))
    if st.shadow:
        d.text((x + st.shadow[0], y + st.shadow[1]), text, font=st.font, fill=(0, 0, 0, 160))
    d.text((x, y), text, font=st.font, fill=st.fill + (255,), stroke_width=st.stroke, stroke_fill=st.stroke_fill + (255,))
    a = np.asarray(im).astype(np.float32) / 255
    a[..., 3] *= st.alpha
    return a


def overlay_mask(ov, size):
    """Hole mask the phone would produce: overlay support, dilated ~12% of the line height (StrokeMasker)."""
    m = (ov[..., 3] > 0.05).astype(np.uint8)
    k = max(3, int(round(size * 0.12)))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))


def compose(frame_rgb01, ov):
    a = ov[..., 3:4]
    return frame_rgb01 * (1 - a) + ov[..., :3] * a


def prep(videos, out_dir, max_short=1080, stride=1):
    """Decode videos into JPEG frame folders (random access during training)."""
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        cap = cv2.VideoCapture(v)
        i = k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % stride == 0:
                h, w = f.shape[:2]
                s = min(1.0, max_short / min(h, w))
                if s < 1:
                    f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(d, '%06d.jpg' % k), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                k += 1
            i += 1
        print(v, '->', d, k, 'frames')


class SubtitleClips(Dataset):
    """Yields (gt, masked_in, mask) as float tensors [T,3,H,W] / [T,1,H,W]; T = nb local consecutive frames + nr
    reference frames sampled further away (same layout as phone inference)."""

    def __init__(self, root, fonts, nb=10, nr=6, length=100000, scale=(0.8, 1.25), seed=None):
        self.clips = [d for d in sorted(glob.glob(os.path.join(root, '*'))) if os.path.isdir(d)]
        self.frames = {d: sorted(glob.glob(os.path.join(d, '*.jpg'))) for d in self.clips}
        self.clips = [d for d in self.clips if len(self.frames[d]) >= nb + 2]
        assert self.clips, 'no frame folders under ' + root
        self.fonts, self.nb, self.nr, self.length, self.scale, self.seed = fonts, nb, nr, length, scale, seed

    def __len__(self):
        return self.length

    def _ids(self, rng, n):
        s = rng.randint(0, n - self.nb)
        local = list(range(s, s + self.nb))
        c = s + self.nb // 2
        refs = []
        for _ in range(self.nr):
            off = rng.choice([-1, 1]) * rng.randint(15, 120)
            refs.append(int(np.clip(c + off, 0, n - 1)))
        return local, refs

    def __getitem__(self, idx):
        rng = random.Random(None if self.seed is None else self.seed * 1000003 + idx)
        d = rng.choice(self.clips)
        fs = self.frames[d]
        local, refs = self._ids(rng, len(fs))
        imgs = [cv2.imread(fs[i])[..., ::-1] for i in local + refs]
        h, w = imgs[0].shape[:2]
        s = rng.uniform(*self.scale) if rng.random() < 0.5 else 1.0
        cw, ch = min(w, round(W_IN * s)), min(h, round(H_IN * s))
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        flip = rng.random() < 0.5
        gt = []
        for im in imgs:
            c = im[y0:y0 + ch, x0:x0 + cw]
            if c.shape[1] != W_IN or c.shape[0] != H_IN:
                c = cv2.resize(c, (W_IN, H_IN), interpolation=cv2.INTER_AREA if s > 1 else cv2.INTER_CUBIC)
            if flip:
                c = c[:, ::-1]
            gt.append(c.astype(np.float32) / 255)
        # subtitles: one line in the local window (switching to another line with p=0.3), own lines for refs
        st = SubtitleStyle(rng, self.fonts)
        cx = W_IN / 2 + rng.uniform(-60, 60)
        cy = rng.uniform(st.size * 0.8, H_IN - st.size * 0.8)
        lines = [random_line(rng)]
        switch = rng.randint(1, self.nb - 1) if rng.random() < 0.3 else None
        if switch:
            lines.append(random_line(rng))
        ov_cache = {}

        def ov_for(key, text):
            if key not in ov_cache:
                ov_cache[key] = render_line(text, st, W_IN, H_IN, cx, cy)
            return ov_cache[key]
        inp, msk = [], []
        for t in range(len(gt)):
            if t < self.nb:
                k = 1 if switch and t >= switch else 0
                ov = ov_for(('l', k), lines[k])
            else:
                ov = ov_for(('r', t), random_line(rng)) if rng.random() < 0.8 else ov_for(('l', 0), lines[0])
            m = overlay_mask(ov, st.size)
            if rng.random() < 0.1:                   # detector-box style hole
                ys, xs = np.nonzero(m)
                if len(xs):
                    m[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5] = 1
            inp.append(compose(gt[t], ov))
            msk.append(m)
        T = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a))).permute(0, 3, 1, 2).float()
        gt_t, in_t = T(gt), T(inp)
        m_t = torch.from_numpy(np.stack(msk)).float()[:, None]
        return gt_t, in_t * (1 - m_t), m_t


def default_fonts():
    fs = [p for p in glob.glob(os.path.join(os.path.dirname(__file__), 'fonts', '*')) if p.lower().endswith(('.otf', '.ttf', '.ttc'))]
    fs += [p for p in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf']
           if os.path.exists(p)]
    return fs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep', help='decode clean videos into frame folders')
    p.add_argument('out')
    p.add_argument('videos', nargs='+')
    p.add_argument('--max-short', type=int, default=1080)
    p.add_argument('--stride', type=int, default=1)
    p = sub.add_parser('preview', help='write a contact sheet of synthetic samples')
    p.add_argument('root')
    p.add_argument('--fonts', nargs='*')
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--out', default='synth_preview.jpg')
    a = ap.parse_args()
    if a.cmd == 'prep':
        prep(a.videos, a.out, a.max_short, a.stride)
    else:
        ds = SubtitleClips(a.root, a.fonts or default_fonts(), seed=1)
        rows = []
        for i in range(a.n):
            gt, inp, m = ds[i]
            row = [gt[0], inp[0] + m[0] * torch.tensor([1., 0, 0])[:, None, None], gt[12]]
            rows.append(np.hstack([(r.permute(1, 2, 0).numpy()[..., ::-1] * 255).clip(0, 255).astype(np.uint8) for r in row]))
        cv2.imwrite(a.out, np.vstack(rows))
        print('wrote', a.out)
