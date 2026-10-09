# Fine-tune the official STTN (teacher) on synthetic subtitles at native resolution, or distil it into a slimmer
# student that fits a phone GPU/NPU budget.
#   finetune: python train.py --mode finetune --data frames/ --teacher sttn.pth --out runs/teacher
#   distill : python train.py --mode distill  --data frames/ --teacher runs/teacher/G_latest.pth --arch m --out runs/m
import os, time, argparse, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

import net, synth

ARCH = {   # name: (channel, stack, encoder widths)
    'teacher': (256, 8, (64, 128)),
    'm': (128, 4, (64, 128)),
    's': (128, 4, (32, 64)),
    'xs': (96, 4, (32, 64)),
}


class VGGLoss(nn.Module):
    """Perceptual + style loss on VGG16 relu1_2..relu3_3; counteracts the L1 tendency to blur texture."""

    def __init__(self):
        super().__init__()
        import torchvision
        v = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in v.parameters():
            p.requires_grad_(False)
        self.v = v
        self.cut = {3, 8, 15}
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def feats(self, x):
        x = ((x + 1) / 2 - self.mean) / self.std
        out = []
        for i, l in enumerate(self.v):
            x = l(x)
            if i in self.cut:
                out.append(x)
        return out

    def forward(self, a, b):
        fa, fb = self.feats(a), self.feats(b)
        perc = sum(F.l1_loss(x, y) for x, y in zip(fa, fb))

        def gram(f):
            n, c, h, w = f.shape
            f = f.reshape(n, c, h * w)
            return f @ f.transpose(1, 2) / (c * h * w)
        style = sum(F.l1_loss(gram(x), gram(y)) for x, y in zip(fa, fb))
        return perc, style


def hf_loss(a, b, m):
    """L1 on Laplacian (high-frequency) inside the hole: directly penalises the blur/grid STTN leaves."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=a.dtype, device=a.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    la, lb = F.conv2d(a, k, padding=1, groups=3), F.conv2d(b, k, padding=1, groups=3)
    return (torch.abs(la - lb) * m).sum() / (m.sum() * 3 + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['finetune', 'distill'], required=True)
    ap.add_argument('--data', required=True, help='folder of frame folders (synth.py prep)')
    ap.add_argument('--teacher', required=True, help='official sttn.pth or a fine-tuned teacher checkpoint')
    ap.add_argument('--arch', default='m', choices=list(ARCH))
    ap.add_argument('--out', required=True)
    ap.add_argument('--fonts', nargs='*')
    ap.add_argument('--iters', type=int, default=100000)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--nb', type=int, default=10)
    ap.add_argument('--nr', type=int, default=6)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--w-hole', type=float, default=1.0)
    ap.add_argument('--w-valid', type=float, default=1.0)
    ap.add_argument('--w-adv', type=float, default=0.01)
    ap.add_argument('--w-perc', type=float, default=0.05)
    ap.add_argument('--w-style', type=float, default=40.0)
    ap.add_argument('--w-hf', type=float, default=1.0)
    ap.add_argument('--w-kd', type=float, default=1.0, help='distill: L1 to teacher output inside the hole')
    ap.add_argument('--no-vgg', action='store_true')
    ap.add_argument('--amp', action='store_true', help='bf16 autocast (GPU)')
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--log-every', type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher = net.load_any(a.teacher).to(dev)
    assert (teacher.channel, teacher.stack) == (256, 8), 'teacher must be the full 256x8 model'
    if a.mode == 'finetune':
        G = teacher
        teacher = None
        arch = ARCH['teacher']
    else:
        arch = ARCH[a.arch]
        G = net.InpaintGenerator(*arch).to(dev)
        net.init_student_from_teacher(G, teacher)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    D = net.Discriminator().to(dev)
    lr = a.lr or (5e-5 if a.mode == 'finetune' else 2e-4)
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.99))
    oD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))
    sched = lambda it: 0.1 if it > a.iters * 0.8 else 1.0
    start = 0
    ck = os.path.join(a.out, 'state_latest.pth')
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        G.load_state_dict(s['G']); D.load_state_dict(s['D']); oG.load_state_dict(s['oG']); oD.load_state_dict(s['oD'])
        start = s['it']
        print('resumed at', start)
    vgg = None
    if not a.no_vgg and (a.w_perc > 0 or a.w_style > 0):
        try:
            vgg = VGGLoss().to(dev)
        except Exception as e:
            print('VGG loss disabled:', e)

    ds = synth.SubtitleClips(a.data, a.fonts or synth.default_fonts(), nb=a.nb, nr=a.nr, length=a.iters * a.bs)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0,
                    pin_memory=dev.type == 'cuda')
    T = a.nb + a.nr
    it = start
    t0 = time.time()
    acc = {}
    amp = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp)
    G.train(); D.train()
    for gt, inp, m in dl:
        if it >= a.iters:
            break
        it += 1
        for g in oG.param_groups + oD.param_groups:
            g['lr'] = lr * sched(it)
        B = gt.shape[0]
        gt = gt.to(dev, non_blocking=True).view(B * T, 3, net.H_IN, net.W_IN) * 2 - 1
        inp = inp.to(dev, non_blocking=True).view(B * T, 3, net.H_IN, net.W_IN) * 2 - 1
        m = m.to(dev, non_blocking=True).view(B * T, 1, net.H_IN, net.W_IN)
        x = inp * (1 - m)
        with amp:
            pred = G(x, T)
        pred = pred.float()
        comp = gt * (1 - m) + pred * m

        with amp:
            d_real, d_fake = D(gt, T), D(comp.detach(), T)
        lD = (F.relu(1 - d_real.float()).mean() + F.relu(1 + d_fake.float()).mean()) / 2
        oD.zero_grad(set_to_none=True)
        lD.backward()
        oD.step()

        L = {}
        with amp:
            L['adv'] = -D(comp, T).float().mean() * a.w_adv
        L['hole'] = (torch.abs(pred - gt) * m).sum() / (m.sum() * 3 + 1) * a.w_hole
        L['valid'] = (torch.abs(pred - gt) * (1 - m)).sum() / ((1 - m).sum() * 3 + 1) * a.w_valid
        L['hf'] = hf_loss(pred, gt, m) * a.w_hf
        if vgg is not None:
            with amp:
                perc, style = vgg(comp, gt)
            L['perc'] = perc.float() * a.w_perc
            L['style'] = style.float() * a.w_style
        if teacher is not None and a.w_kd > 0:
            with torch.no_grad(), amp:
                tp = teacher(x, T).float()
            L['kd'] = (torch.abs(pred - tp) * m).sum() / (m.sum() * 3 + 1) * a.w_kd
        lG = sum(L.values())
        oG.zero_grad(set_to_none=True)
        lG.backward()
        oG.step()

        L['D'] = lD
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        if it % a.log_every == 0 or it == start + 1:
            n = 1 if it == start + 1 else a.log_every
            print('it %d  %.2fs/it  ' % (it, (time.time() - t0) / n) + '  '.join('%s %.4f' % (k, v / n) for k, v in acc.items()), flush=True)
            acc = {}
            t0 = time.time()
        if it % a.save_every == 0 or it == a.iters:
            torch.save({'netG': G.state_dict(), 'arch': arch}, os.path.join(a.out, 'G_latest.pth'))
            torch.save({'netG': G.state_dict(), 'arch': arch}, os.path.join(a.out, 'G_%06d.pth' % it))
            torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'oG': oG.state_dict(), 'oD': oD.state_dict(), 'it': it}, ck)
    torch.save({'netG': G.state_dict(), 'arch': arch}, os.path.join(a.out, 'G_latest.pth'))
    print('done', it)


if __name__ == '__main__':
    main()
