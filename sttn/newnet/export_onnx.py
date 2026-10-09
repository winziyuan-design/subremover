# Export FGFI-Net for the phone and check it.
#   python sttn/newnet/export_onnx.py runs/fgfi/G_latest.pth --h 204 --w 1080 --t 8 [--fp16] --out app/assets/fgfi_204x1080.onnx
# Shapes are static (NNAPI needs them): export one file per band size the App uses (H % 12 == 0, W % 72 == 0;
# the App grows the band's context to reach those multiples instead of padding inside the graph).
# Checks: onnxruntime vs torch parity, op histogram, ops outside the NNAPI-friendly set, max tensor rank.
import os, sys, argparse, collections, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FGFINet, PhoneWrapper, load, align, CFG

# ops ORT's NNAPI EP has builders for (subset we may emit). Anything else falls back to the CPU provider.
NNAPI_OK = {'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 'Add', 'Sub', 'Mul', 'Div', 'Relu', 'LeakyRelu', 'Sigmoid', 'Tanh',
            'Softmax', 'Reshape', 'Transpose', 'Concat', 'Split', 'Slice', 'Resize', 'MaxPool', 'AveragePool', 'Clip',
            'Exp', 'Sqrt', 'Pow', 'ReduceMean', 'Squeeze', 'Unsqueeze', 'Identity', 'Cast', 'Gather', 'Flatten', 'Neg', 'Abs'}


def dummy_inputs(T, H, W, seed=0):
    r = np.random.RandomState(seed)
    fr = r.rand(T, 3, H, W).astype(np.float32)
    m = np.zeros((T, 2, H, W), np.float32)
    y0 = H // 3
    for t in range(T):
        m[t, 0, y0:y0 + H // 4, W // 8 + 4 * t: W - W // 8] = 1
        m[t, 1, y0 + H // 4: y0 + H // 3, W // 4: W // 2] = 1
    ff = (r.randn(T - 1, 2, H // 2, W // 2) * 2).astype(np.float32)
    fb = -ff
    return fr, m, ff, fb


def export(g, path, T, H, W, fp16=False, opset=17):
    g = g.eval()
    g.set_export(True)
    w = PhoneWrapper(g).eval()
    args = tuple(torch.from_numpy(a) for a in dummy_inputs(T, H, W))
    with torch.no_grad():
        torch.onnx.export(w, args, path, input_names=['frames', 'masks', 'flow_fw', 'flow_bw'], output_names=['out'],
                          opset_version=opset, dynamo=False, do_constant_folding=True)
    # fold the (static) shape arithmetic the tracer leaves behind; result is provider-independent (basic level)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = path
    ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    if fp16:
        import onnx
        from onnxconverter_common import float16
        mdl = float16.convert_float_to_float16(onnx.load(path), keep_io_types=True, op_block_list=['GridSample'])
        onnx.save(mdl, path)
    return w


def inspect(path):
    import onnx
    mdl = onnx.shape_inference.infer_shapes(onnx.load(path))
    hist = collections.Counter(n.op_type for n in mdl.graph.node)
    rank = 0
    for vi in list(mdl.graph.value_info) + list(mdl.graph.input) + list(mdl.graph.output):
        if vi.type.tensor_type.HasField('shape'):
            rank = max(rank, len(vi.type.tensor_type.shape.dim))
    return hist, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='?', help='checkpoint from train.py; omit with --cfg to export random weights (tests)')
    ap.add_argument('--cfg', default='base', choices=list(CFG))
    ap.add_argument('--t', type=int, default=8)
    ap.add_argument('--h', type=int, default=204)
    ap.add_argument('--w', type=int, default=1080)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default='fgfi.onnx')
    a = ap.parse_args()
    g = load(a.ckpt) if a.ckpt else FGFINet(a.cfg)
    ah, aw = align(g.cfg)
    assert a.h % ah == 0 and a.w % aw == 0, 'H must be a multiple of %d and W of %d' % (ah, aw)
    w = export(g, a.out, a.t, a.h, a.w, a.fp16)
    hist, rank = inspect(a.out)
    print('wrote %s  %.1f MB  max tensor rank %d' % (a.out, os.path.getsize(a.out) / 1e6, rank))
    print('ops:', dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    print('ops outside the NNAPI set (run on CPU EP):', sorted(set(hist) - NNAPI_OK) or 'none')
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    s = ort.InferenceSession(a.out, so, providers=['CPUExecutionProvider'])
    inp = dummy_inputs(a.t, a.h, a.w, seed=1)
    feed = dict(zip(['frames', 'masks', 'flow_fw', 'flow_bw'], inp))
    s.run(None, feed)
    t0 = time.time()
    o = s.run(None, feed)[0]
    dt = time.time() - t0
    with torch.no_grad():
        ref = w(*[torch.from_numpy(x) for x in inp]).numpy()
    d = np.abs(o - ref)
    print('parity vs torch: max |diff| %.2e  mean %.2e   ORT CPU %d threads: %.2f s / window' % (d.max(), d.mean(), a.threads, dt))


if __name__ == '__main__':
    main()
