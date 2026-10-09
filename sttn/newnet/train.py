# Train FGFI-Net on synthetic subtitles / watermarks (data.py). Single GPU:
#   python sttn/newnet/train.py --data frames/train --val-data frames/val --out runs/fgfi --stage 1
# Multi-GPU (DDP):  torchrun --nproc_per_node 4 sttn/newnet/train.py ... (same args; --bs is per GPU)
# Stages (see README / docs/new-net-design.md): 1 reconstruction warm-up (no GAN), 2 full losses + hard augmentation,
# 3 larger crops / lower lr. Each stage resumes from --init (previous stage's G_latest.pth) or its own state_latest.pth.
import os, sys, time, json, math, argparse, copy, multiprocessing as mp
import torch, torch.nn.functional as F, torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FGFINet, CFG, align, count_params  # noqa: E402
import data, losses, metrics  # noqa: E402

STAGES = {   # defaults per stage; any explicit CLI value wins
    1: dict(crop='192x432', iters=40000, lr=2e-4, w_adv=0.0, w_perc=0.0, w_style=0.0, w_temp=0.0, level0=0.2, level1=0.6),
    2: dict(crop='192x432', iters=120000, lr=1e-4, w_adv=0.01, w_perc=0.05, w_style=40.0, w_temp=0.5, level0=0.6, level1=1.0),
    3: dict(crop='240x864', iters=40000, lr=3e-5, w_adv=0.01, w_perc=0.05, w_style=40.0, w_temp=0.5, level0=1.0, level1=1.0),
}


def ddp_setup():
    if 'RANK' not in os.environ:
        return 0, 1, 0
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    lr_ = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(lr_)
    return dist.get_rank(), dist.get_world_size(), lr_


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def flat(x):
    return x.reshape(-1, *x.shape[2:])


@torch.no_grad()
def validate(G, dl, dev, T, amp, n_batches):
    G.eval()
    acc = {'psnr': 0.0, 'ssim': 0.0, 'detail': 0.0}
    n = 0
    for i, b in enumerate(dl):
        if i >= n_batches:
            break
        b = to_dev(b, dev)
        gt, inp, m = flat(b['gt']), flat(b['inp']), flat(b['masks'])
        with amp:
            pred = G(inp * 2 - 1, m, flat(b['ff']), flat(b['fb']), T)
        hole = torch.clamp(m[:, 0:1] + m[:, 1:2], 0, 1)
        comp = gt * (1 - hole) + ((pred.float() + 1) / 2).clamp(0, 1) * hole
        acc['psnr'] += metrics.psnr_hole(comp, gt, hole)
        acc['ssim'] += metrics.ssim_hole(comp, gt, hole)
        acc['detail'] += metrics.detail_ratio(comp, gt, hole)
        n += 1
    G.train()
    return {k: v / max(1, n) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder of frame folders (sttn/train/synth.py prep)')
    ap.add_argument('--val-data', help='held-out frame folders (different source videos!)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', type=int, default=1, choices=[1, 2, 3])
    ap.add_argument('--init', help='start G (and D if present) from this checkpoint, e.g. previous stage G_latest.pth')
    ap.add_argument('--cfg', default='base', choices=list(CFG))
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--crop')
    ap.add_argument('--iters', type=int)
    ap.add_argument('--bs', type=int, default=4, help='clips per GPU')
    ap.add_argument('--lr', type=float)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--fonts', nargs='*')
    ap.add_argument('--w-hole', type=float, default=1.0)
    ap.add_argument('--w-valid', type=float, default=0.5)
    ap.add_argument('--w-hf', type=float, default=1.0)
    ap.add_argument('--w-flow', type=float, default=0.25)
    ap.add_argument('--w-adv', type=float)
    ap.add_argument('--w-perc', type=float)
    ap.add_argument('--w-style', type=float)
    ap.add_argument('--w-temp', type=float)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--bf16', action='store_true', help='bf16 autocast (A100/H100/4090/L40...)')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--val-every', type=int, default=2000)
    ap.add_argument('--val-batches', type=int, default=25)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--no-vgg', action='store_true')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    st = STAGES[a.stage]
    for k, v in st.items():
        if k in ('level0', 'level1'):
            continue
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    H, W = map(int, a.crop.split('x'))

    rank, world, local = ddp_setup()
    dev = torch.device(a.device or ('cuda:%d' % local if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(1234 + rank)
    main_ = rank == 0
    if main_:
        os.makedirs(a.out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(a.out, 'args_stage%d.json' % a.stage), 'w'), indent=1)

    G = FGFINet(a.cfg).to(dev)
    ah, aw = align(G.cfg)
    assert H % ah == 0 and W % aw == 0, 'crop must be H%%%d==0 and W%%%d==0' % (ah, aw)
    D = losses.Discriminator().to(dev)
    G_ema = copy.deepcopy(G).eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)
    oG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    oD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.0, 0.99))
    start = 0
    ck = os.path.join(a.out, 'state_stage%d.pth' % a.stage)
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        G.load_state_dict(s['G']); G_ema.load_state_dict(s['G_ema']); D.load_state_dict(s['D'])
        oG.load_state_dict(s['oG']); oD.load_state_dict(s['oD'])
        start = s['it']
        if main_:
            print('resumed stage %d at it %d' % (a.stage, start))
    elif a.init:
        s = torch.load(a.init, map_location=dev, weights_only=False)
        G.load_state_dict(s['G']); G_ema.load_state_dict(s.get('G_ema', s['G']))
        if 'D' in s:
            D.load_state_dict(s['D'])
        if main_:
            print('initialised from', a.init)
    if main_:
        print('FGFI-Net %s: %.2fM params, D %.2fM, crop %dx%d, T %d, bs %d x %d GPU(s)' % (
            a.cfg, count_params(G) / 1e6, count_params(D) / 1e6, H, W, a.T, a.bs, world))
    Gd, Dd = G, D
    if world > 1:
        Gd = torch.nn.parallel.DistributedDataParallel(G, device_ids=[local] if dev.type == 'cuda' else None)
        Dd = torch.nn.parallel.DistributedDataParallel(D, device_ids=[local] if dev.type == 'cuda' else None)

    vgg = None
    if not a.no_vgg and (a.w_perc > 0 or a.w_style > 0):
        vgg = losses.VGGLoss().to(dev)

    level = mp.Value('d', st['level0'])
    ds = data.Clips(a.data, a.fonts, T=a.T, crop=(H, W), length=(a.iters - start + 10) * a.bs * world,
                    seed=None, level=level)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=dev.type == 'cuda',
                    persistent_workers=a.workers > 0, worker_init_fn=data.worker_init, prefetch_factor=4 if a.workers else None)
    vdl = None
    if a.val_data and main_:
        vds = data.Clips(a.val_data, a.fonts, T=a.T, crop=(H, W), length=a.val_batches * a.bs, seed=777,
                         level=mp.Value('d', 1.0), gt_flow=False)
        vdl = DataLoader(vds, batch_size=a.bs, num_workers=min(4, a.workers), worker_init_fn=data.worker_init)

    amp = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.bf16)
    T = a.T
    it, t0, acc, best = start, time.time(), {}, -1.0
    G.train(); D.train()
    use_gan = a.w_adv > 0
    for b in dl:
        if it >= a.iters:
            break
        it += 1
        frac = it / a.iters
        level.value = st['level0'] + (st['level1'] - st['level0']) * min(1.0, frac / 0.7)
        lr = max(a.lr * 0.02, a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))) * min(1.0, it / min(1000.0, 0.05 * a.iters))  # warm-up + cosine
        for g in oG.param_groups + oD.param_groups:
            g['lr'] = lr
        b = to_dev(b, dev)
        gt, inp, m = flat(b['gt']) * 2 - 1, flat(b['inp']) * 2 - 1, flat(b['masks'])
        ff, fb, gff, gfb = flat(b['ff']), flat(b['fb']), flat(b['gff']), flat(b['gfb'])
        hole = torch.clamp(m[:, 0:1] + m[:, 1:2], 0, 1)
        with amp:
            pred, fc, bc = Gd(inp, m, ff, fb, T, return_flow=True)
        pred, fc, bc = pred.float(), fc.float(), bc.float()
        comp = gt * (1 - hole) + pred * hole

        if use_gan:
            with amp:
                d_out = Dd(torch.cat([gt, comp.detach()], 0), T).float()      # one forward (DDP-safe)
                nb = d_out.shape[0] // 2
                lD = losses.d_hinge(d_out[:nb], d_out[nb:])
            oD.zero_grad(set_to_none=True)
            lD.backward()
            oD.step()

        L = {'hole': losses.masked_l1(pred, gt, hole) * a.w_hole,
             'valid': losses.masked_l1(pred, gt, 1 - hole) * a.w_valid,
             'hf': losses.hf_loss(pred, gt, hole) * a.w_hf,
             'flow': losses.flow_loss(fc, bc, gff, gfb, F.max_pool2d(hole, 4), T) * a.w_flow}
        if use_gan:
            with amp:
                L['adv'] = losses.g_hinge(Dd(comp, T).float()) * a.w_adv
        if vgg is not None:
            with amp:
                perc, style = vgg(comp, gt)
            L['perc'] = perc.float() * a.w_perc
            L['style'] = style.float() * a.w_style
        if a.w_temp > 0:
            L['temp'] = losses.temporal_loss(comp, gff, gfb, hole, T) * a.w_temp
        lG = sum(L.values())
        oG.zero_grad(set_to_none=True)
        lG.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), a.grad_clip)
        oG.step()
        with torch.no_grad():
            dcy = min(a.ema, (1 + it) / (10 + it))
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.mul_(dcy).add_(p.detach(), alpha=1 - dcy)
            for be, b_ in zip(G_ema.buffers(), G.buffers()):
                be.copy_(b_)
        if use_gan:
            L['D'] = lD
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())

        if main_ and (it % a.log_every == 0 or it == start + 1):
            n = 1 if it == start + 1 else a.log_every
            mem = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == 'cuda' else 0
            print('it %d  %.3fs/it  lr %.2e  level %.2f  mem %.1fG  ' % (it, (time.time() - t0) / n, lr, level.value, mem)
                  + '  '.join('%s %.4f' % (k, v / n) for k, v in acc.items()), flush=True)
            acc, t0 = {}, time.time()
        last = it == a.iters
        if main_ and (it % a.save_every == 0 or last):
            meta = {'cfg': G.cfg, 'it': it, 'stage': a.stage}
            torch.save(dict(meta, G=G.state_dict(), G_ema=G_ema.state_dict(), D=D.state_dict()), os.path.join(a.out, 'G_latest.pth'))
            torch.save(dict(meta, G=G.state_dict(), G_ema=G_ema.state_dict()), os.path.join(a.out, 'G_s%d_%06d.pth' % (a.stage, it)))
            torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(), 'D': D.state_dict(), 'oG': oG.state_dict(),
                        'oD': oD.state_dict(), 'it': it}, ck + '.tmp')
            os.replace(ck + '.tmp', ck)
        if main_ and vdl is not None and (it % a.val_every == 0 or last):
            v = validate(G_ema, vdl, dev, T, amp, a.val_batches)
            print('VAL it %d  PSNR(hole) %.2f  SSIM(hole) %.4f  detail %.3f' % (it, v['psnr'], v['ssim'], v['detail']), flush=True)
            with open(os.path.join(a.out, 'val.jsonl'), 'a') as f:
                f.write(json.dumps(dict(v, it=it, stage=a.stage)) + '\n')
            if v['psnr'] > best:
                best = v['psnr']
                torch.save({'cfg': G.cfg, 'it': it, 'stage': a.stage, 'G': G.state_dict(), 'G_ema': G_ema.state_dict(), 'val': v},
                           os.path.join(a.out, 'G_best_s%d.pth' % a.stage))
        if world > 1 and it % a.save_every == 0:
            dist.barrier()
    if main_:
        print('done stage %d at it %d' % (a.stage, it))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
