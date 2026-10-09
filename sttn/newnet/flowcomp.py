# Light flow completion. Input: flows the phone already has (DIS on the band at 1/2 res, hole filled by normalised
# convolution exactly like app FlowField) taken to 1/4 res. A small dilated U-Net predicts a residual inside the
# (dilated) hole for the forward and backward flow of each adjacent pair at once (they constrain each other).
import torch, torch.nn as nn, torch.nn.functional as F

from ops import conv, ResBlock, up2


class FlowCompletion(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.e1 = conv(2 + 2 + 2, c)
        self.e2 = conv(c, 2 * c, s=2)
        self.mid = nn.Sequential(ResBlock(2 * c, 1), ResBlock(2 * c, 2), ResBlock(2 * c, 4), ResBlock(2 * c, 8))
        self.d1 = conv(2 * c, c)
        self.d2 = conv(2 * c, c)
        self.out = nn.Conv2d(c, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f, b, ma, mb):
        """f: a->b flow on grid a, b: b->a flow on grid b ([N,2,h,w], 1/4-res pixels); ma/mb holes of a/b ([N,1,h,w]).
        Returns completed (f, b). Flow is scaled by 1/8 inside the net to keep activations O(1)."""
        x = torch.cat([f * 0.125, b * 0.125, ma, mb], 1)
        e1 = self.e1(x)
        y = self.mid(self.e2(e1))
        y = self.d2(torch.cat([self.d1(up2(y, e1)), e1], 1))
        d = self.out(y) * 8.0
        sa = F.max_pool2d(ma, 5, 1, 2)
        sb = F.max_pool2d(mb, 5, 1, 2)
        return f + d[:, 0:2] * sa, b + d[:, 2:4] * sb
