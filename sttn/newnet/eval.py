# Evaluation protocol (docs/new-net-design.md section 7).
#  synth: held-out clean clips + synthetic subtitles (+ optional watermark) at native resolution; in-hole PSNR / SSIM /
#         LPIPS (optional) / detail-energy ratio, for FGFI-Net and the STTN baselines (official weights on native tiles,
#         subtitle-finetuned teacher on native tiles) through the same harness.
#    python sttn/newnet/eval.py synth --frames frames/val --fgfi runs/fgfi/G_best_s3.pth \
#        --sttn sttn.pth --teacher models/teacher_G_latest.pth --out eval_synth
#  real:  our sample a.mp4 at the four reference shots, side by side with STTN tiles, teacher tiles and the
#         mini-program output b.mp4 (no ground truth: sheets for eyes + detail energy vs b.mp4).
#    python sttn/newnet/eval.py real --video a.mp4 --ref b.mp4 --masks sttn/newnet/realmasks \
#        --fgfi runs/fgfi/G_best_s3.pth --sttn sttn.pth --teacher models/teacher_G_latest.pth --out eval_real
import os, sys, glob, json, random, argparse, time
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'train'))
import infer, metrics, data  # noqa: E402
import synth, eval_tiles  # noqa: E402  (sttn/train)

SHOTS = {'glass_3.8s': 115, 'poster_12s': 355, 'bowl_18s': 535, 'packets_28s': 835}   # first frame of an 8-frame window


def load_methods(a, T):
    ms = {}
    if a.fgfi:
        r = infer.Runner(a.fgfi, threads=a.threads)
        ms['fgfi'] = lambda fr, mk, rect: infer.run_window(r, fr, mk, fixed=r.fixed)
    for name, ck in (('sttn_tiles', a.sttn), ('teacher_tiles', a.teacher)):
        if ck:
            mdl = eval_tiles.Model(ck, T)

            def f(fr, mk, rect, mdl=mdl):
                outs = [x.copy() for x in fr]
                for (x0, y0, x1, y1) in infer.regions(mk):
                    o, _ = eval_tiles.run_tiles(mdl, outs, [(m > 0).astype(np.uint8) * 255 for m in mk], (x0, y0, x1, y1))
                    outs = [np.where((m > 0)[..., None], oo, x) for oo, x, m in zip(o, outs, mk)]
                return outs
            ms[name] = f
    return ms


def to_t(ims):
    return torch.from_numpy(np.stack(ims).astype(np.float32) / 255).permute(0, 3, 1, 2)


def cmd_synth(a):
    T = a.T
    fonts = data.find_fonts(a.fonts)
    ms = load_methods(a, T)
    lp = None
    if a.lpips:
        lp = metrics.LPIPSHole()
    rng = random.Random(7)
    res = {k: [] for k in ms}
    sheet = []
    for clip in sorted(d for d in glob.glob(os.path.join(a.frames, '*')) if os.path.isdir(d)):
        fs = sorted(glob.glob(os.path.join(clip, '*.jpg')) + glob.glob(os.path.join(clip, '*.png')))
        if len(fs) < T + 2:
            continue
        for wi in range(a.windows):
            s = int((wi + 0.5) * (len(fs) - T) / a.windows)
            gt = [cv2.imread(fs[s + t])[..., ::-1].copy() for t in range(T)]
            H, W = gt[0].shape[:2]
            st = synth.SubtitleStyle(rng, fonts)
            st.size, st.box, st.alpha = max(20, int(min(H, W) * 0.045)), False, 1.0
            st.font = synth.ImageFont.truetype(st.font_path, st.size)
            line = ''
            while len(line) < 12:
                line += synth.random_line(rng)
            line = line[:14]
            ovs = [synth.render_line(line, st, W, H, W / 2, H * 0.8)] * T
            wm = None
            if a.watermark:
                wst = synth.SubtitleStyle(rng, fonts)
                wst.alpha, wst.box, wst.size = 0.6, False, max(16, int(min(H, W) * 0.03))
                wst.font = synth.ImageFont.truetype(wst.font_path, wst.size)
                txt = data.wm_text(rng)
                wm = [synth.render_line(txt, wst, W, H, W * 0.3 + 4 * t, H * 0.35) for t in range(T)]
            inp, msk = [], []
            for t in range(T):
                x = synth.compose(gt[t].astype(np.float32) / 255, ovs[t])
                m = synth.overlay_mask(ovs[t], st.size)
                if wm:
                    x = synth.compose(x, wm[t])
                    m |= synth.overlay_mask(wm[t], wst.size)
                inp.append((x * 255 + 0.5).clip(0, 255).astype(np.uint8))
                msk.append(m)
            row = [inp[T // 2], gt[T // 2]]
            for k, f in ms.items():
                t0 = time.time()
                out = f(inp, msk, None)
                dt = time.time() - t0
                o, g, mm = to_t(out), to_t(gt), torch.from_numpy(np.stack(msk).astype(np.float32))[:, None]
                r = dict(psnr=metrics.psnr_hole(o, g, mm), ssim=metrics.ssim_hole(o, g, mm),
                         detail=metrics.detail_ratio(o, g, mm), sec=dt)
                if lp:
                    r['lpips'] = lp(o, g, mm)
                res[k].append(r)
                row.append(out[T // 2])
            ys, xs = np.nonzero(msk[T // 2])
            y0, y1 = max(0, ys.min() - 40), min(H, ys.max() + 40)
            x0, x1 = max(0, xs.min() - 40), min(W, xs.max() + 40)
            sheet.append(np.vstack([r_[y0:y1, x0:x1] for r_ in row]))
            print(os.path.basename(clip), wi, {k: '%.2f' % v[-1]['psnr'] for k, v in res.items()}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    summ = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in res.items() if v}
    print('\nmethod         PSNR(hole)  SSIM(hole)  detail  ' + ('LPIPS  ' if a.lpips else '') + 'sec/window')
    for k, s in summ.items():
        print('%-14s %8.2f   %8.4f   %6.3f  ' % (k, s['psnr'], s['ssim'], s['detail']) + ('%6.4f ' % s['lpips'] if a.lpips else '') + '%6.1f' % s['sec'])
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    if sheet:
        wmax = max(s.shape[1] for s in sheet)
        cv2.imwrite(os.path.join(a.out, 'sheet.jpg'), np.vstack([cv2.copyMakeBorder(s, 0, 8, 0, wmax - s.shape[1], cv2.BORDER_CONSTANT)
                                                              for s in sheet[:6]])[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('sheet rows per sample: input, ground truth, ' + ', '.join(ms))


def read_frames(path, idx, size=None):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx[0])
    out = []
    for _ in idx:
        ok, f = cap.read()
        assert ok, 'cannot read %s frame' % path
        if size and (f.shape[1], f.shape[0]) != size:
            f = cv2.resize(f, size, interpolation=cv2.INTER_CUBIC)
        out.append(f[..., ::-1].copy())
    return out


def lap_energy(im, m):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float((cv2.Laplacian(g, cv2.CV_32F)[m > 0] ** 2).mean())


def cmd_real(a):
    T = a.T
    ms = load_methods(a, T)
    os.makedirs(a.out, exist_ok=True)
    shots = SHOTS if not a.shots else {s.split('=')[0]: int(s.split('=')[1]) for s in a.shots}
    report = {}
    for name, s0 in shots.items():
        idx = list(range(s0, s0 + T))
        fr = read_frames(a.video, idx)
        H, W = fr[0].shape[:2]
        mk = []
        for i in idx:
            p = sorted(glob.glob(os.path.join(a.masks, '*_mask_%04d.png' % i)) + glob.glob(os.path.join(a.masks, '%04d.png' % i)))
            assert p, 'no mask for frame %d in %s' % (i, a.masks)
            mk.append((cv2.imread(p[0], 0) > 0).astype(np.uint8))
        cols = {'input': fr}
        for k, f in ms.items():
            t0 = time.time()
            cols[k] = f(fr, mk, None)
            print(name, k, '%.1fs' % (time.time() - t0), flush=True)
        if a.ref:
            cols['miniprogram(b.mp4)'] = read_frames(a.ref, idx, (W, H))
        q = T // 2
        rep = {}
        if a.ref:
            ref_e = lap_energy(cols['miniprogram(b.mp4)'][q], mk[q])
            rep = {k: lap_energy(v[q], mk[q]) / max(ref_e, 1e-6) for k, v in cols.items() if k != 'input'}
        report[name] = rep
        # crops: the subtitle band (lowest region) and the moving watermark region, full width
        tiles = []
        for (x0, y0, x1, y1) in sorted(infer.regions(mk), key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))[:2]:
            yy0, yy1 = max(0, y0 - 60), min(H, y1 + 60)
            xx0, xx1 = max(0, x0 - 60), min(W, x1 + 60)
            strip = []
            for k, v in cols.items():
                c = v[q][yy0:yy1, xx0:xx1].copy()
                cv2.putText(c, k, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                strip.append(c)
            tiles.append(np.hstack(strip) if (xx1 - xx0) < (yy1 - yy0) else np.vstack(strip))
        for j, t_ in enumerate(tiles):
            cv2.imwrite(os.path.join(a.out, '%s_region%d.png' % (name, j)), t_[..., ::-1])
        print(name, 'detail energy vs b.mp4 (1.0 = as sharp as mini-program):', {k: round(v, 2) for k, v in rep.items()}, flush=True)
    json.dump(report, open(os.path.join(a.out, 'detail_vs_b.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for n in ('synth', 'real'):
        p = sub.add_parser(n)
        p.add_argument('--fgfi', help='FGFI-Net checkpoint (.pth) or exported .onnx')
        p.add_argument('--sttn', help='official STTN sttn.pth (native-res tiles baseline)')
        p.add_argument('--teacher', help='subtitle-finetuned STTN teacher (models/teacher_G_latest.pth)')
        p.add_argument('--T', type=int, default=8)
        p.add_argument('--threads', type=int, default=4)
        p.add_argument('--out', default='eval_' + n)
    p = sub.choices['synth']
    p.add_argument('--frames', required=True, help='held-out clean frame folders')
    p.add_argument('--fonts', nargs='*')
    p.add_argument('--windows', type=int, default=3)
    p.add_argument('--watermark', action='store_true')
    p.add_argument('--lpips', action='store_true')
    p = sub.choices['real']
    p.add_argument('--video', required=True)
    p.add_argument('--ref', help='mini-program output (b.mp4)')
    p.add_argument('--masks', required=True, help='folder with pipe.py masks (<x>_mask_%%04d.png) or %%04d.png')
    p.add_argument('--shots', nargs='*', help='name=first_frame ... (default: the four reference shots)')
    a = ap.parse_args()
    (cmd_synth if a.cmd == 'synth' else cmd_real)(a)


if __name__ == '__main__':
    main()
