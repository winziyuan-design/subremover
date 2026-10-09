# Small building blocks shared by the generator and the flow-completion net. Everything here stays at rank <= 4 and
# uses ops ONNX Runtime's NNAPI / XNNPACK / CPU providers know (Conv, ConvTranspose, MatMul, Softmax, Resize, elementwise),
# except GridSample (flow warping), which runs on the CPU provider. Written from the papers, no third-party code.
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

_GRIDS = {}


def act():
    return nn.LeakyReLU(0.2, inplace=True)


def conv(cin, cout, k=3, s=1, d=1, a=True):
    m = [nn.Conv2d(cin, cout, k, s, padding=d * (k // 2), dilation=d)]
    if a:
        m.append(act())
    return nn.Sequential(*m)


class ResBlock(nn.Module):
    def __init__(self, c, d=1):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=d, dilation=d)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.leaky_relu(self.c1(x), 0.2))


def up2(x, like=None):
    """Nearest x2 upsample (ONNX Resize); with `like`, to exactly that map's size (odd sizes)."""
    if like is None:
        return F.interpolate(x, scale_factor=2.0, mode='nearest')
    return F.interpolate(x, size=(int(like.shape[-2]), int(like.shape[-1])), mode='nearest')


def base_grid(h, w, ref):
    """[1,2,h,w] pixel coordinates (x, y)."""
    h, w = int(h), int(w)
    key = (h, w, ref.device, ref.dtype)
    g = _GRIDS.get(key)
    if g is None:
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        g = torch.from_numpy(np.stack([xs, ys])[None]).to(device=ref.device, dtype=ref.dtype)   # constant in ONNX
        _GRIDS[key] = g
    return g


def warp(x, flow):
    """Backward warp: out(p) = x(p + flow(p)). x [N,C,h,w], flow [N,2,h,w] in pixels of this resolution."""
    h, w = int(x.shape[2]), int(x.shape[3])
    g = base_grid(h, w, flow) + flow
    gx = g[:, 0:1] * (2.0 / max(w - 1, 1)) - 1.0
    gy = g[:, 1:2] * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.cat([gx, gy], 1).permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)


def fb_valid(f, b):
    """Soft forward-backward consistency of flow f (a->b, on grid a) against b (b->a, on grid b). 1 = consistent."""
    bw = warp(b, f)
    sq = lambda z: z[:, 0:1] * z[:, 0:1] + z[:, 1:2] * z[:, 1:2]
    err = sq(f + bw)
    mag = sq(f) + sq(bw)
    return torch.exp(-err / (0.01 * mag + 0.5))


class ChannelNorm(nn.Module):
    """LayerNorm over channels of a [N,C,h,w] map written with elementwise ops (no LayerNormalization node)."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(1, keepdim=True)
        return xc / torch.sqrt(var + self.eps) * self.g + self.b
