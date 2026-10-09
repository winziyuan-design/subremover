# FGFI-Net generator (docs/new-net-design.md). Resolution-agnostic: conv encoder/decoder at native resolution,
# completed-flow-guided bidirectional feature propagation at 1/4, then a soft-split transformer whose tokens are
# overlapping 7x7 patches (stride 3) of the 1/4 map, attending inside spatio-temporal windows (all T frames x full strip
# height x `ws` token columns, shifted every other block) with completed flow embedded into queries/keys.
# Soft split / soft composition follow FuseFormer (Liu et al., ICCV 2021): unfold+Linear == Conv2d(k7,s3) and
# Linear+fold == ConvTranspose2d(k7,s3), normalised by the overlap count. Flow guidance follows the ideas of FGT
# (Zhang et al., ECCV 2022) and E2FGVI (Li et al., CVPR 2022). Own implementation; no code copied from those repos.
import math, torch, torch.nn as nn, torch.nn.functional as F

from ops import conv, ResBlock, up2, warp, fb_valid, ChannelNorm, act
from flowcomp import FlowCompletion

CFG = {   # c1/c2/c4: encoder widths at 1, 1/2, 1/4; d: token dim; cm: channels of the soft-composed map in F3N
    'tiny': dict(c1=16, c2=24, c4=32, d=64, heads=2, depth=2, cm=8, ws=6, fc=16),
    'small': dict(c1=16, c2=32, c4=64, d=128, heads=4, depth=6, cm=24, ws=6, fc=24),
    'base': dict(c1=24, c2=48, c4=96, d=192, heads=4, depth=8, cm=32, ws=6, fc=32),
}
K, S, P = 7, 3, 3          # soft split kernel / stride / padding at 1/4 resolution -> one token per 12x12 input px


def align(cfg):
    """(H multiple, W multiple) the generator needs: H/4 divisible by 3, token columns divisible by the window."""
    return 4 * S, 4 * S * cfg['ws']


class SoftComp(nn.Module):
    """Linear + fold == ConvTranspose2d(k7,s3), divided by how many patches cover each position."""

    def __init__(self, cin, cout):
        super().__init__()
        self.t = nn.ConvTranspose2d(cin, cout, K, S, P, output_padding=S - 1)

    def forward(self, x):
        y = self.t(x)
        return y * overlap_inv(int(x.shape[2]), int(x.shape[3]), x)


_OV = {}


def overlap_inv(h, w, ref):
    """1 / (number of 7x7 stride-3 patches covering each position of the composed map), as a constant."""
    key = (h, w, ref.device, ref.dtype)
    if key not in _OV:
        with torch.no_grad():
            c = F.conv_transpose2d(torch.ones(1, 1, h, w), torch.ones(1, 1, K, K), None, S, P, S - 1)
        _OV[key] = torch.from_numpy((1.0 / c).numpy()).to(device=ref.device, dtype=ref.dtype)
    return _OV[key]


class Propagation(nn.Module):
    """Bidirectional recurrent propagation at 1/4: warp the neighbour's propagated feature with the completed flow,
    gate it by forward-backward consistency, fuse residually (no deformable conv: not exportable to phone delegates)."""

    def __init__(self, c):
        super().__init__()
        self.bwd = nn.Sequential(conv(2 * c + 4, c), ResBlock(c), nn.Conv2d(c, c, 3, padding=1))
        self.fwd = nn.Sequential(conv(2 * c + 4, c), ResBlock(c), nn.Conv2d(c, c, 3, padding=1))
        self.fuse = nn.Conv2d(2 * c, c, 1)

    @staticmethod
    def _step(body, cur, nb, flow, valid, m):
        w = warp(nb, flow) * valid
        return cur + body(torch.cat([cur, w, valid, m, flow * 0.125], 1))

    def forward(self, feats, ff, fb, vf, vb, m):
        """lists over time: feats[t], m[t]; ff[t] t->t+1 (grid t), fb[t] t+1->t (grid t+1), vf/vb their validity."""
        T = len(feats)
        hb = [None] * T
        hb[T - 1] = feats[T - 1]
        for t in range(T - 2, -1, -1):
            hb[t] = self._step(self.bwd, feats[t], hb[t + 1], ff[t], vf[t], m[t])
        hf = [hb[0]]
        for t in range(1, T):
            hf.append(self._step(self.fwd, hb[t], hf[t - 1], fb[t - 1], vb[t - 1], m[t]))
        return [feats[t] + self.fuse(torch.cat([hb[t], hf[t]], 1)) for t in range(T)]


class WindowAttention(nn.Module):
    def __init__(self, d, heads, ws):
        super().__init__()
        self.h, self.ws = heads, ws
        self.qk = nn.Conv2d(2 * d, 2 * d, 1)      # tokens + flow embedding -> q, k
        self.v = nn.Conv2d(d, d, 1)
        self.proj = nn.Conv2d(d, d, 1)
        self.explicit = False                     # True for ONNX export (MatMul/Softmax instead of SDPA)

    def _part(self, x, B, T):
        """[B*T,d,h,w] -> [B*nw, heads, T*h*ws, dh] (rank <= 4 throughout)."""
        bt, d, h, w = (int(v) for v in x.shape)
        nw = w // self.ws
        x = x.permute(0, 2, 3, 1).reshape(B, T * h, nw, self.ws * d).permute(0, 2, 1, 3)
        x = x.reshape(B * nw, T * h * self.ws, self.h, d // self.h).permute(0, 2, 1, 3)
        return x

    def _unpart(self, x, B, T, d, h, w):
        nw = w // self.ws
        x = x.permute(0, 2, 1, 3).reshape(B, nw, T * h, self.ws * d).permute(0, 2, 1, 3)
        return x.reshape(B * T, h, w, d).permute(0, 3, 1, 2)

    def forward(self, x, femb, B, T, shift):
        bt, d, h, w = (int(v) for v in x.shape)
        qk = self.qk(torch.cat([x, femb], 1))
        q, k = qk[:, :d], qk[:, d:]
        v = self.v(x)
        if shift:
            q, k, v = (torch.roll(z, -shift, 3) for z in (q, k, v))
        q, k, v = (self._part(z, B, T) for z in (q, k, v))
        if self.explicit:
            a = torch.softmax(torch.matmul(q * (1.0 / math.sqrt(int(q.shape[-1]))), k.transpose(-2, -1)), -1)
            o = torch.matmul(a, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v)
        o = self._unpart(o, B, T, d, h, w)
        if shift:
            o = torch.roll(o, shift, 3)
        return self.proj(o)


class F3N(nn.Module):
    """Fusion feed-forward: tokens -> soft composition (overlapping fold) -> soft split -> tokens, so neighbouring
    overlapping patches exchange sub-token detail (removes the block artefacts of hard patch splitting)."""

    def __init__(self, d, cm):
        super().__init__()
        self.comp = SoftComp(d, cm)
        self.split = nn.Conv2d(cm, d, K, S, P)

    def forward(self, x):
        return self.split(F.leaky_relu(self.comp(x), 0.2))


class Block(nn.Module):
    def __init__(self, d, heads, cm, ws, shift):
        super().__init__()
        self.pos = nn.Conv2d(d, d, 3, padding=1, groups=d)        # conditional position encoding (res-agnostic)
        self.n1, self.n2 = ChannelNorm(d), ChannelNorm(d)
        self.attn = WindowAttention(d, heads, ws)
        self.ffn = F3N(d, cm)
        self.shift = shift

    def forward(self, x, femb, B, T):
        x = x + self.pos(x)
        x = x + self.attn(self.n1(x), femb, B, T, self.shift)
        return x + self.ffn(self.n2(x))


class FGFINet(nn.Module):
    def __init__(self, cfg='base'):
        super().__init__()
        c = CFG[cfg] if isinstance(cfg, str) else cfg
        self.cfg = dict(c)
        c1, c2, c4, d = c['c1'], c['c2'], c['c4'], c['d']
        self.fc = FlowCompletion(c['fc'])
        self.e1 = conv(3 + 2, c1)
        self.e2 = nn.Sequential(conv(c1, c2, s=2), conv(c2, c2))
        self.e4 = nn.Sequential(conv(c2, c4, s=2), conv(c4, c4), ResBlock(c4))
        self.prop = Propagation(c4)
        self.split = nn.Conv2d(c4, d, K, S, P)                     # soft split (overlapping 7x7 patches, stride 3)
        self.femb = nn.Conv2d(2 + 2 + 2 + 1, d, K, S, P)           # flow / validity / mask embedding per token
        self.blocks = nn.ModuleList(Block(d, c['heads'], c['cm'], c['ws'], (i % 2) * (c['ws'] // 2))
                                    for i in range(c['depth']))
        self.comp = SoftComp(d, c4)                                # soft composition back to the 1/4 map
        self.d4 = nn.Sequential(conv(2 * c4, c4), ResBlock(c4))
        self.d2a = conv(c4, c2)
        self.d2b = nn.Sequential(conv(2 * c2, c2), ResBlock(c2))
        self.d1a = conv(c2, c1)
        self.d1b = nn.Sequential(conv(2 * c1, c1), nn.Conv2d(c1, 3, 3, padding=1))

    def set_export(self, on=True):
        for b in self.blocks:
            b.attn.explicit = on

    def complete_flow(self, ff2, fb2, m4, B, T):
        """half-res flows [B*(T-1),2,H/2,W/2] -> completed 1/4-res flows (same layout) + validity."""
        f4 = F.avg_pool2d(ff2, 2) * 0.5
        b4 = F.avg_pool2d(fb2, 2) * 0.5
        h, w = int(m4.shape[2]), int(m4.shape[3])
        mm = m4.reshape(B, T, h, w)
        ma = mm[:, :T - 1].reshape(B * (T - 1), 1, h, w)
        mb = mm[:, 1:].reshape(B * (T - 1), 1, h, w)
        fc, bc = self.fc(f4, b4, ma, mb)
        return fc, bc, fb_valid(fc, bc), fb_valid(bc, fc)

    def forward(self, x, masks, ff2, fb2, T, return_flow=False):
        """x [B*T,3,H,W] in [-1,1] (unknown hole pixels zeroed, prefilled pixels kept); masks [B*T,2,H,W]
        (0: still unknown, 1: prefilled by the App's pixel propagation); ff2/fb2 [B*(T-1),2,H/2,W/2] DIS flows
        (t->t+1 and t+1->t) in half-res pixels. Returns RGB in [-1,1] for every pixel (composite outside)."""
        bt, H, W = int(x.shape[0]), int(x.shape[2]), int(x.shape[3])
        B = bt // T
        ah, aw = align(self.cfg)
        assert H % ah == 0 and W % aw == 0, 'pad/choose the band so H %% %d == 0 and W %% %d == 0' % (ah, aw)
        hole = torch.clamp(masks[:, 0:1] + masks[:, 1:2], 0, 1)
        m4 = F.max_pool2d(hole, 4)
        h4, w4 = int(m4.shape[2]), int(m4.shape[3])
        fc, bc, vf, vb = self.complete_flow(ff2, fb2, m4, B, T)

        s1 = self.e1(torch.cat([x, masks], 1))
        s2 = self.e2(s1)
        f4 = self.e4(s2)
        c4 = int(f4.shape[1])

        def split_t(z, n):
            c = int(z.shape[1])
            z = z.reshape(B, n * c, h4, w4)
            return [z[:, i * c:(i + 1) * c] for i in range(n)]
        L = lambda z: split_t(z, T)
        Lp = lambda z: split_t(z, T - 1)
        feats, ms = L(f4), L(m4)
        ffl, fbl, vfl, vbl = Lp(fc), Lp(bc), Lp(vf), Lp(vb)
        g = self.prop(feats, ffl, fbl, vfl, vbl, ms)
        g = torch.cat(g, 1).reshape(B * T, c4, h4, w4)

        z2, z1 = torch.zeros_like(ffl[0]), torch.zeros_like(vfl[0])
        fe = [torch.cat([ffl[t] if t < T - 1 else z2, fbl[t - 1] if t > 0 else z2,
                         vfl[t] if t < T - 1 else z1, vbl[t - 1] if t > 0 else z1], 1) for t in range(T)]
        fe = torch.cat(fe, 1).reshape(B * T, 6, h4, w4)
        femb = self.femb(torch.cat([fe[:, 0:4] * 0.125, fe[:, 4:6], m4], 1))

        tok = self.split(g)
        for blk in self.blocks:
            tok = blk(tok, femb, B, T)
        y = self.d4(torch.cat([self.comp(tok), g], 1))
        y = self.d2b(torch.cat([self.d2a(up2(y, s2)), s2], 1))
        y = self.d1b(torch.cat([self.d1a(up2(y, s1)), s1], 1))
        out = torch.tanh(y)
        if return_flow:
            return out, fc, bc
        return out


class PhoneWrapper(nn.Module):
    """ONNX entry: frames [T,3,H,W] RGB 0..1, masks [T,2,H,W] (unknown, prefilled), flow_fw / flow_bw [T-1,2,H/2,W/2]
    (DIS, half-res pixels, as app FlowField computes them) -> out [T,3,H,W] RGB 0..1, composited outside the hole."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, frames, masks, flow_fw, flow_bw):
        T = int(frames.shape[0])
        hole = torch.clamp(masks[:, 0:1] + masks[:, 1:2], 0, 1)
        x = (frames * 2 - 1) * (1 - masks[:, 0:1])
        o = (self.g(x, masks, flow_fw, flow_bw, T) + 1) * 0.5
        return frames * (1 - hole) + o * hole


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load(path, map_location='cpu', ema=True):
    s = torch.load(path, map_location=map_location, weights_only=False)
    g = FGFINet(s['cfg'])
    key = 'G_ema' if ema and 'G_ema' in s else 'G'
    g.load_state_dict(s[key])
    return g
