# STTN generator / T-PatchGAN discriminator (researchmm/STTN, MIT), made width/depth configurable so the same code
# loads the official 256x8 teacher and builds slimmer phone students. Parameter names match the official checkpoint.
import math, torch, torch.nn as nn, torch.nn.functional as F

W_IN, H_IN = 432, 240                                  # fixed by the patch sizes below (feature map 108x60)
PATCHES = [(108, 60), (36, 20), (18, 10), (9, 5)]


class Attention(nn.Module):
    def forward(self, q, k, v):
        # the official model computes masked_fill out-of-place and drops the result, i.e. it never masks;
        # keep that so the released weights behave identically
        p = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)
        return torch.matmul(p, v)


class MultiHeadedAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query_embedding = nn.Conv2d(d, d, 1)
        self.value_embedding = nn.Conv2d(d, d, 1)
        self.key_embedding = nn.Conv2d(d, d, 1)
        self.output_linear = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, t):
        bt, c, h, w = x.shape
        b = bt // t
        dk = c // len(PATCHES)
        out = []
        qs = torch.chunk(self.query_embedding(x), len(PATCHES), 1)
        ks = torch.chunk(self.key_embedding(x), len(PATCHES), 1)
        vs = torch.chunk(self.value_embedding(x), len(PATCHES), 1)
        for (pw, ph), q, k, v in zip(PATCHES, qs, ks, vs):
            oh, ow = h // ph, w // pw
            y = self.attention(_tok(q, b, oh, ph, ow, pw), _tok(k, b, oh, ph, ow, pw), _tok(v, b, oh, ph, ow, pw))
            out.append(_untok(y, bt, dk, oh, ph, ow, pw))
        return self.output_linear(torch.cat(out, 1))


# Patch (un)folding kept at rank <= 4 so NNAPI / GPU delegates can take the whole graph
# (the official 7-D reshape/permute would be split off to the CPU).
def _tok(z, b, oh, ph, ow, pw):
    bt, dk = z.shape[0], z.shape[1]
    z = z.reshape(bt * dk * oh, ph, ow, pw).permute(0, 2, 1, 3)               # [bt*dk*oh, ow, ph, pw]
    z = z.reshape(bt, dk, oh * ow, ph * pw).permute(0, 2, 1, 3)               # [bt, oh*ow, dk, ph*pw]
    return z.reshape(b, (bt // b) * oh * ow, dk * ph * pw)


def _untok(y, bt, dk, oh, ph, ow, pw):
    y = y.reshape(bt, oh * ow, dk, ph * pw).permute(0, 2, 1, 3)               # [bt, dk, oh*ow, ph*pw]
    y = y.reshape(bt * dk * oh, ow, ph, pw).permute(0, 2, 1, 3)               # [bt*dk*oh, ph, ow, pw]
    return y.reshape(bt, dk, oh * ph, ow * pw)


class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(d, d, 3, padding=2, dilation=2), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(d, d, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = MultiHeadedAttention(d)
        self.feed_forward = FeedForward(d)

    def forward(self, x, t):
        x = x + self.attention(x, t)
        return x + self.feed_forward(x)


class Deconv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class InpaintGenerator(nn.Module):
    """channel must be divisible by 4 (one head per patch size). Official teacher: channel=256, stack=8."""

    def __init__(self, channel=256, stack=8, enc=(64, 128)):
        super().__init__()
        e1, e2 = enc
        self.channel, self.stack = channel, stack
        self.encoder = nn.Sequential(
            nn.Conv2d(3, e1, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(e1, e1, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(e1, e2, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(e2, channel, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.transformer = nn.ModuleList([TransformerBlock(channel) for _ in range(stack)])
        self.decoder = nn.Sequential(
            Deconv(channel, e2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(e2, e1, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            Deconv(e1, e1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(e1, 3, 3, 1, 1))

    def forward(self, x, t, return_feats=False):
        """x: [B*T,3,240,432] in [-1,1] with holes zeroed. Returns [B*T,3,240,432] in [-1,1]."""
        f = self.encoder(x)
        feats = []
        for blk in self.transformer:
            f = blk(f, t)
            feats.append(f)
        o = torch.tanh(self.decoder(f))
        return (o, feats) if return_feats else o


def load_teacher(path):
    g = InpaintGenerator(256, 8)
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('netG', sd)
    g.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    return g


def load_any(path):
    """Official sttn.pth or a checkpoint written by train.py (which records its arch)."""
    sd = torch.load(path, map_location='cpu')
    arch = sd.get('arch', (256, 8, (64, 128)))
    g = InpaintGenerator(arch[0], arch[1], tuple(arch[2]))
    g.load_state_dict({k.replace('module.', ''): v for k, v in sd.get('netG', sd).items()})
    return g


def init_student_from_teacher(student, teacher):
    """Copy what fits: encoder/decoder outer layers, and for each student block the evenly spaced teacher block,
    channel-sliced. A head start, not a substitute for distillation."""
    s_sd, t_sd = student.state_dict(), teacher.state_dict()
    pick = [round(i * (teacher.stack - 1) / max(1, student.stack - 1)) for i in range(student.stack)]
    for k, v in s_sd.items():
        tk = k
        if k.startswith('transformer.'):
            parts = k.split('.')
            parts[1] = str(pick[int(parts[1])])
            tk = '.'.join(parts)
        tv = t_sd.get(tk)
        if tv is None or tv.dim() != v.dim():
            continue
        sl = tuple(slice(0, n) for n in v.shape)
        s_sd[k] = tv[sl].clone() if all(a <= b for a, b in zip(v.shape, tv.shape)) else v
    student.load_state_dict(s_sd)


class Discriminator(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.conv = nn.Sequential(
            sn(nn.Conv3d(3, nf, (3, 5, 5), (1, 2, 2), padding=1, bias=False)), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv3d(nf, nf * 2, (3, 5, 5), (1, 2, 2), padding=(1, 2, 2), bias=False)), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv3d(nf * 2, nf * 4, (3, 5, 5), (1, 2, 2), padding=(1, 2, 2), bias=False)), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv3d(nf * 4, nf * 4, (3, 5, 5), (1, 2, 2), padding=(1, 2, 2), bias=False)), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv3d(nf * 4, nf * 4, (3, 5, 5), (1, 2, 2), padding=(1, 2, 2), bias=False)), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, (3, 5, 5), (1, 2, 2), padding=(1, 2, 2)))

    def forward(self, x, t):
        """x: [B*T,3,H,W] -> [B,C,T',H',W']"""
        bt, c, h, w = x.shape
        return self.conv(x.reshape(bt // t, t, c, h, w).transpose(1, 2))


class PhoneWrapper(nn.Module):
    """ONNX entry used on the phone: frames [T,3,240,432] RGB 0..1, masks [T,1,240,432] 1=hole -> out RGB 0..1."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, frames, masks):
        t = frames.shape[0]
        o = self.g((frames * 2 - 1) * (1 - masks), t)
        return (o + 1) / 2


def gflops_per_frame(channel, stack, enc=(64, 128)):
    """Multiply-adds x2 per output frame at 432x240 (attention matmuls excluded; they are small here)."""
    e1, e2 = enc
    h2, w2, h4, w4 = 120, 216, 60, 108
    conv = lambda ci, co, k, h, w: 2 * ci * co * k * k * h * w
    f = conv(3, e1, 3, h2, w2) + conv(e1, e1, 3, h2, w2) + conv(e1, e2, 3, h4, w4) + conv(e2, channel, 3, h4, w4)
    f += stack * (3 * conv(channel, channel, 1, h4, w4) + 3 * conv(channel, channel, 3, h4, w4))
    f += conv(channel, e2, 3, h2, w2) + conv(e2, e1, 3, h2, w2) + conv(e1, e1, 3, 240, 432) + conv(e1, 3, 3, 240, 432)
    return f / 1e9
