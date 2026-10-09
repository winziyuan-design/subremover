# Ground-truth comparison on synthetic subtitles over clean HD footage:
#   down  = current pipe.py layout: context around the hole squeezed into one 432x240 canvas, result upscaled
#   tiles = native-resolution 432x240 tiles over the hole (overlap + feathering), as the phone now does
# Reports PSNR and detail (Laplacian energy ratio vs ground truth) inside the hole.
#   python eval_tiles.py --ckpt sttn.pth --frames frames/ [--ckpt2 runs/s/G_latest.pth]
import argparse, glob, os, random, time
import cv2, numpy as np, onnxruntime as ort, torch

import net, synth

TW, TH = net.W_IN, net.H_IN


def tile_grid(x0, y0, x1, y1, W, H, margin=24, overlap=48):
    """Tile origins covering [x0,x1)x[y0,y1) (+margin context) with 432x240 tiles at native scale."""
    def axis(a0, a1, size, full):
        a0, a1 = max(0, a0 - margin), min(full, a1 + margin)
        span = a1 - a0
        if span <= size:
            c = (a0 + a1) // 2
            return [int(np.clip(c - size // 2, 0, max(0, full - size)))]
        n = int(np.ceil((span - overlap) / (size - overlap)))
        step = (span - size) / (n - 1)
        return [int(np.clip(round(a0 + i * step), 0, full - size)) for i in range(n)]
    return [(x, y) for y in axis(y0, y1, TH, H) for x in axis(x0, x1, TW, W)]


def feather(W, H, tiles, ramp=24):
    """Per-tile blend weights that fall off towards inner tile edges (not at image borders)."""
    ws = []
    for x, y in tiles:
        wx = np.ones(TW, np.float32)
        wy = np.ones(TH, np.float32)
        r = np.linspace(0.05, 1, ramp, dtype=np.float32)
        if x > 0: wx[:ramp] = r
        if x + TW < W: wx[-ramp:] = r[::-1]
        if y > 0: wy[:ramp] = r
        if y + TH < H: wy[-ramp:] = r[::-1]
        ws.append(wy[:, None] * wx[None])
    return ws


class Model:
    def __init__(self, ckpt, T):
        p = '/tmp/eval_%s_t%d.onnx' % (os.path.basename(os.path.dirname(ckpt)) + os.path.basename(ckpt), T)
        if not os.path.exists(p):
            g = net.PhoneWrapper(net.load_any(ckpt).eval())
            with torch.no_grad():
                torch.onnx.export(g, (torch.rand(T, 3, TH, TW), torch.zeros(T, 1, TH, TW)), p,
                                  input_names=['frames', 'masks'], output_names=['out'], opset_version=17, dynamo=False)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.s = ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
        self.T, self.calls, self.sec = T, 0, 0.0

    def __call__(self, frames, masks):
        """frames list of HxWx3 uint8 RGB (432x240), masks list of HxW uint8 -> list of uint8 RGB"""
        fr = np.stack(frames).astype(np.float32).transpose(0, 3, 1, 2) / 255
        mk = (np.stack(masks)[:, None] > 0).astype(np.float32)
        t0 = time.time()
        o = self.s.run(None, {'frames': fr, 'masks': mk})[0]
        self.sec += time.time() - t0
        self.calls += 1
        return list((np.clip(o.transpose(0, 2, 3, 1), 0, 1) * 255 + 0.5).astype(np.uint8))


def run_down(model, frames, masks, rect):
    """frames full-res RGB, masks full-res; one canvas holding the context rect at 432:240 aspect."""
    H, W = masks[0].shape
    x0, y0, x1, y1 = rect
    asp = TW / TH
    pw, ph = x1 - x0, y1 - y0
    cw, ch = (pw, int(np.ceil(pw / asp))) if pw / ph > asp else (int(np.ceil(ph * asp)), ph)
    cw, ch = min(cw, W), min(ch, H)
    bx = int(np.clip((x0 + x1) // 2 - cw // 2, 0, W - cw))
    by = int(np.clip((y0 + y1) // 2 - ch // 2, 0, H - ch))
    fr = [cv2.resize(f[by:by + ch, bx:bx + cw], (TW, TH), interpolation=cv2.INTER_AREA) for f in frames]
    mk = [cv2.dilate(cv2.resize(m[by:by + ch, bx:bx + cw], (TW, TH), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8)) for m in masks]
    out = model(fr, mk)
    res = []
    for f, o in zip(frames, out):
        r = f.copy()
        r[by:by + ch, bx:bx + cw] = cv2.resize(o, (cw, ch), interpolation=cv2.INTER_CUBIC)
        res.append(r)
    return res, cw / TW


def run_tiles(model, frames, masks, rect):
    H, W = masks[0].shape
    tiles = tile_grid(*rect, W, H)
    ws = feather(W, H, tiles)
    acc = [np.zeros((H, W, 3), np.float32) for _ in frames]
    wsum = np.zeros((H, W), np.float32)
    for (x, y), w in zip(tiles, ws):
        out = model([f[y:y + TH, x:x + TW] for f in frames], [m[y:y + TH, x:x + TW] for m in masks])
        for k, o in enumerate(out):
            acc[k][y:y + TH, x:x + TW] += o.astype(np.float32) * w[..., None]
        wsum[y:y + TH, x:x + TW] += w
    res = []
    for f, a in zip(frames, acc):
        r = f.copy()
        cov = wsum > 0
        r[cov] = (a[cov] / wsum[cov, None] + 0.5).clip(0, 255).astype(np.uint8)
        res.append(r)
    return res, len(tiles)


def metrics(out, gt, hole):
    h = hole > 0
    mse = ((out.astype(np.float32) - gt.astype(np.float32))[h] ** 2).mean()
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)
    detail = (lap(out)[h] ** 2).mean() / max(1e-6, (lap(gt)[h] ** 2).mean())
    return 10 * np.log10(255 ** 2 / max(mse, 1e-6)), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ckpt2', help='optional second model (e.g. fine-tuned / student) evaluated with tiles')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--fonts', nargs='*')
    ap.add_argument('--nb', type=int, default=8)
    ap.add_argument('--nr', type=int, default=4)
    ap.add_argument('--windows', type=int, default=3, help='test windows per clip')
    ap.add_argument('--out', default='eval_tiles')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fonts = a.fonts or synth.default_fonts()
    T = a.nb + a.nr
    models = {'down': Model(a.ckpt, T), 'tiles': Model(a.ckpt, T)}
    if a.ckpt2:
        models['tiles2'] = Model(a.ckpt2, T)
    rng = random.Random(7)
    res = {k: [] for k in models}
    sheet = []
    for clip in sorted(d for d in glob.glob(os.path.join(a.frames, '*')) if os.path.isdir(d)):
        fs = sorted(glob.glob(os.path.join(clip, '*.jpg')))
        for wi in range(a.windows):
            s = int((wi + 0.5) * (len(fs) - a.nb) / a.windows)
            local = list(range(s, s + a.nb))
            refs = [int(np.clip(s + a.nb // 2 + o, 0, len(fs) - 1)) for o in (-60, -30, 30, 60)][:a.nr]
            gt = [cv2.imread(fs[i])[..., ::-1].copy() for i in local + refs]
            H, W = gt[0].shape[:2]
            st = synth.SubtitleStyle(rng, fonts)
            st.size, st.box, st.alpha = int(H * 0.05), False, 1.0
            st.font = synth.ImageFont.truetype(st.font_path, st.size)
            cy = int(H * 0.8)
            line = synth.random_line(rng)
            while len(line) < 10:
                line += synth.random_line(rng)
            line = line[:14]
            inp, msk = [], []
            for t in range(T):
                ov = synth.render_line(line if t < a.nb else synth.random_line(rng), st, W, H, W / 2, cy)
                m = synth.overlay_mask(ov, st.size)
                inp.append((synth.compose(gt[t].astype(np.float32) / 255, ov) * 255 + 0.5).clip(0, 255).astype(np.uint8))
                msk.append(m * 255)
            U = np.zeros((H, W), np.uint8)
            for m in msk[:a.nb]:
                U |= m
            ys, xs = np.nonzero(U)
            rect = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            outs = {}
            for k, mdl in models.items():
                if k == 'down':
                    mg = 12
                    r = (max(0, rect[0] - mg), max(0, rect[1] - mg), min(W, rect[2] + mg), min(H, rect[3] + mg))
                    o, info = run_down(mdl, inp, msk, r)
                else:
                    o, info = run_tiles(mdl, inp, msk, rect)
                comp = []
                for t in range(a.nb):
                    soft = cv2.GaussianBlur((msk[t] > 0).astype(np.float32), (5, 5), 0)
                    soft = np.maximum(soft, (msk[t] > 0).astype(np.float32))[..., None]
                    comp.append((o[t] * soft + inp[t] * (1 - soft) + 0.5).astype(np.uint8))
                    res[k].append(metrics(comp[-1], gt[t], msk[t]))
                outs[k] = (comp, info)
            q = a.nb // 2
            y0c, y1c = max(0, rect[1] - 40), min(H, rect[3] + 40)
            x0c, x1c = max(0, rect[0] - 40), min(W, rect[2] + 40)
            rows = [inp[q][y0c:y1c, x0c:x1c], gt[q][y0c:y1c, x0c:x1c]] + [outs[k][0][q][y0c:y1c, x0c:x1c] for k in models]
            sheet.append(np.vstack(rows))
            print(os.path.basename(clip), 'win', wi, ' '.join('%s(%s)' % (k, ('scale %.2f' % (1 / v[1])) if k == 'down' else '%d tiles' % v[1])
                                                         for k, v in outs.items()), flush=True)
    print('\nrows in sheet: input, ground truth, ' + ', '.join(models))
    for k, v in res.items():
        v = np.array(v)
        print('%-7s PSNR in hole %.2f dB   detail ratio %.2f (1.0 = as sharp as ground truth)   %d calls %.1fs/call' % (
            k, v[:, 0].mean(), np.median(v[:, 1]), models[k].calls, models[k].sec / max(1, models[k].calls)))
    hmax = max(s.shape[0] for s in sheet)
    cv2.imwrite(os.path.join(a.out, 'sheet.jpg'), np.hstack([cv2.copyMakeBorder(s, 0, hmax - s.shape[0], 0, 8, cv2.BORDER_CONSTANT)
                                                          for s in sheet[:3]])[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == '__main__':
    main()
