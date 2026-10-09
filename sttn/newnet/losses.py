# Losses: hole / valid L1, VGG perceptual + style and the Laplacian high-frequency term (reused from sttn/train/train.py),
# T-PatchGAN hinge (discriminator reused from sttn/train/net.py), temporal warping consistency, flow-completion L1.
import os, sys, importlib.util
import torch, torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'train'))
import net as sttn_net  # noqa: E402

_spec = importlib.util.spec_from_file_location('sttn_train_v1', os.path.join(_HERE, '..', 'train', 'train.py'))
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
VGGLoss, hf_loss = _v1.VGGLoss, _v1.hf_loss
Discriminator = sttn_net.Discriminator          # T-PatchGAN (3D conv, spectral norm); works on any crop size >= 64

from ops import warp, fb_valid  # noqa: E402


def masked_l1(a, b, m):
    return (torch.abs(a - b) * m).sum() / (m.sum() * a.shape[1] + 1)


def temporal_loss(comp, gff, gfb, hole, T):
    """|comp_t - warp(comp_{t+1}, gt flow t->t+1)| where the clean flow is forward-backward consistent, weighted to the
    hole of either frame. comp [B*T,3,H,W]; gff/gfb [B*(T-1),2,H/2,W/2] clean-frame flows (half-res px)."""
    BT, C, H, W = comp.shape
    B = BT // T
    f = F.interpolate(gff, size=(H, W), mode='bilinear', align_corners=False) * 2
    b = F.interpolate(gfb, size=(H, W), mode='bilinear', align_corners=False) * 2
    c = comp.reshape(B, T, C, H, W)
    hm = hole.reshape(B, T, 1, H, W)
    a_, n_ = c[:, :-1].reshape(-1, C, H, W), c[:, 1:].reshape(-1, C, H, W)
    w = warp(n_, f)
    with torch.no_grad():
        v = (fb_valid(f, b) > 0.5).float()
        m = torch.clamp(hm[:, :-1].reshape(-1, 1, H, W) + warp(hm[:, 1:].reshape(-1, 1, H, W), f), 0, 1)
        m = F.max_pool2d(m, 9, 1, 4) * v
    return masked_l1(a_, w, m)


def flow_loss(fc, bc, gff, gfb, m4, T):
    """L1 of completed 1/4-res flows vs clean-frame flows, hole region weighted x5."""
    g4f = F.avg_pool2d(gff, 2) * 0.5
    g4b = F.avg_pool2d(gfb, 2) * 0.5
    BT, _, h, w = m4.shape
    B = BT // T
    mm = m4.reshape(B, T, h, w)
    wa = 1 + 4 * mm[:, :-1].reshape(-1, 1, h, w)
    wb = 1 + 4 * mm[:, 1:].reshape(-1, 1, h, w)
    return (torch.abs(fc - g4f) * wa).mean() + (torch.abs(bc - g4b) * wb).mean()


def d_hinge(d_real, d_fake):
    return (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()) / 2


def g_hinge(d_fake):
    return -d_fake.mean()
