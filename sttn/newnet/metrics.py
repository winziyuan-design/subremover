# In-hole metrics shared by train.py validation and eval.py. Images are float tensors in [0,1], [N,3,H,W]; m [N,1,H,W].
import torch, torch.nn.functional as F


def psnr_hole(a, b, m):
    mse = ((a - b) ** 2 * m).sum() / (m.sum() * 3 + 1e-6)
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-10)))


def _gauss(ks=11, sigma=1.5, device='cpu'):
    x = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None]).view(1, 1, ks, ks).repeat(3, 1, 1, 1)


def ssim_hole(a, b, m):
    k = _gauss(device=a.device)
    mu_a, mu_b = F.conv2d(a, k, padding=5, groups=3), F.conv2d(b, k, padding=5, groups=3)
    saa = F.conv2d(a * a, k, padding=5, groups=3) - mu_a ** 2
    sbb = F.conv2d(b * b, k, padding=5, groups=3) - mu_b ** 2
    sab = F.conv2d(a * b, k, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float((s * m).sum() / (m.sum() * 3 + 1e-6))


def detail_ratio(a, b, m):
    """Laplacian energy of a vs b inside the hole (1.0 = as sharp as ground truth; <1 blurry, >1 noisy/ringing)."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=a.dtype, device=a.device).view(1, 1, 3, 3)
    ga, gb = a.mean(1, keepdim=True), b.mean(1, keepdim=True)
    la, lb = F.conv2d(ga, k, padding=1), F.conv2d(gb, k, padding=1)
    return float((la ** 2 * m).sum() / ((lb ** 2 * m).sum() + 1e-8))


class LPIPSHole:
    """Optional LPIPS (pip install lpips; BSD-2, AlexNet ImageNet weights - evaluation only) on hole bounding crops."""

    def __init__(self, device='cpu'):
        import lpips
        self.f = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, a, b, m):
        vals = []
        for i in range(a.shape[0]):
            ys, xs = torch.nonzero(m[i, 0] > 0, as_tuple=True)
            if len(ys) == 0:
                continue
            y0, y1 = max(0, int(ys.min()) - 8), int(ys.max()) + 9
            x0, x1 = max(0, int(xs.min()) - 8), int(xs.max()) + 9
            if y1 - y0 < 32 or x1 - x0 < 32:
                continue
            vals.append(float(self.f(a[i:i + 1, :, y0:y1, x0:x1] * 2 - 1, b[i:i + 1, :, y0:y1, x0:x1] * 2 - 1)))
        return sum(vals) / max(1, len(vals))
