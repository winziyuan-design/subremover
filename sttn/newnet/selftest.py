# CPU self-test (no GPU / data needed): forward+backward of the tiny config, a few optimiser steps on random data,
# params + FLOPs of each config at 8 x (204 x 1080), ONNX export + onnxruntime parity at a small size.
#   python sttn/newnet/selftest.py
import os, sys, time, tempfile
import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FGFINet, CFG, count_params  # noqa: E402
import export_onnx, losses  # noqa: E402


def rand_batch(B, T, H, W):
    x = torch.rand(B * T, 3, H, W) * 2 - 1
    m = torch.zeros(B * T, 2, H, W)
    m[:, 0, H // 3: H // 2, W // 6: W - W // 6] = 1
    m[:, 1, H // 2: H // 2 + 6, W // 4: W // 2] = 1
    ff = torch.randn(B * (T - 1), 2, H // 2, W // 2)
    return x, m, ff, -ff


def main():
    torch.manual_seed(0)
    T, B, H, W = 4, 2, 48, 216
    g = FGFINet('tiny')
    opt = torch.optim.Adam(g.parameters(), 1e-3)
    x, m, ff, fb = rand_batch(B, T, H, W)
    gt = torch.rand_like(x) * 2 - 1
    hole = torch.clamp(m[:, :1] + m[:, 1:], 0, 1)
    ls = []
    for i in range(8):
        pred, fc, bc = g(x * (1 - m[:, :1]), m, ff, fb, T, return_flow=True)
        loss = losses.masked_l1(pred, gt, hole) + losses.masked_l1(pred, gt, 1 - hole) + 0.1 * losses.temporal_loss(
            gt * (1 - hole) + pred * hole, ff, fb, hole, T)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ls.append(float(loss))
    print('tiny fwd/bwd ok; loss over 8 steps on a fixed batch: %.3f -> %.3f' % (ls[0], ls[-1]))
    assert ls[-1] < ls[0]

    for cfg in CFG:
        n = FGFINet(cfg).eval()
        n.set_export(True)
        c = FlopCounterMode(display=False)
        with torch.no_grad(), c:
            n(*rand_batch(1, 8, 204, 1080), 8)
        f = c.get_total_flops()
        print('%-5s %6.2f M params   8x(204x1080): %6.1f GFLOP / window  %5.1f GMAC / frame' % (
            cfg, count_params(n) / 1e6, f / 1e9, f / 2e9 / 8))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 't.onnx')
        w = export_onnx.export(g.eval(), p, T, H, W)
        hist, rank = export_onnx.inspect(p)
        import onnxruntime as ort
        s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
        inp = export_onnx.dummy_inputs(T, H, W, seed=3)
        o = s.run(None, dict(zip(['frames', 'masks', 'flow_fw', 'flow_bw'], inp)))[0]
        with torch.no_grad():
            r = w(*[torch.from_numpy(z) for z in inp]).numpy()
        print('onnx (trained-for-8-steps tiny): max rank %d, non-NNAPI ops %s, parity max |diff| %.2e' % (
            rank, sorted(set(hist) - export_onnx.NNAPI_OK), np.abs(o - r).max()))
        assert rank <= 4 and np.abs(o - r).max() < 1e-4
    print('selftest passed')


if __name__ == '__main__':
    main()
