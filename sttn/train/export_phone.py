# Export a generator checkpoint to the ONNX the Android app loads (assets/sttn.onnx):
#   inputs  frames [T,3,240,432] RGB 0..1, masks [T,1,240,432] 1=hole ; output out [T,3,240,432] RGB 0..1
#   python export_phone.py runs/m/G_latest.pth --t 12 --out sttn.onnx [--fp16]
import argparse, time
import numpy as np, torch, onnx, onnxruntime as ort

import net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--t', type=int, default=12, help='frames per call = local frames (nb) + reference frames (nr)')
    ap.add_argument('--out', default='sttn.onnx')
    ap.add_argument('--fp16', action='store_true', help='also write <out>_fp16.onnx (fp16 weights, fp32 io)')
    ap.add_argument('--bench', type=int, default=2)
    a = ap.parse_args()
    g = net.load_any(a.ckpt).eval()
    w = net.PhoneWrapper(g)
    T = a.t
    torch.manual_seed(0)
    fr = torch.rand(T, 3, net.H_IN, net.W_IN)
    mk = torch.zeros(T, 1, net.H_IN, net.W_IN)
    mk[:, :, 150:200, 100:300] = 1
    with torch.no_grad():
        ref = w(fr, mk).numpy()
        torch.onnx.export(w, (fr, mk), a.out, input_names=['frames', 'masks'], output_names=['out'], opset_version=17,
                          dynamo=False)
    m = onnx.load(a.out)
    m.metadata_props.add(key='sttn_arch', value='%d,%d,%d,%d' % (g.channel, g.stack, g.encoder[0].out_channels, g.encoder[4].out_channels))
    onnx.save(m, a.out)
    m = onnx.shape_inference.infer_shapes(m)
    ranks = [len(v.type.tensor_type.shape.dim) for v in list(m.graph.value_info) + list(m.graph.output)]
    ops = sorted({n.op_type for n in m.graph.node})
    print('arch c=%d stack=%d  %.1f GFLOP/frame  max tensor rank %d' % (g.channel, g.stack, net.gflops_per_frame(
        g.channel, g.stack, (g.encoder[0].out_channels, g.encoder[4].out_channels)), max(ranks)))
    print('ops:', ' '.join(ops))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(a.out, so, providers=['CPUExecutionProvider'])
    o = s.run(None, {'frames': fr.numpy(), 'masks': mk.numpy()})[0]
    print('onnx vs torch maxdiff %.2e' % np.abs(o - ref).max())
    if a.bench:
        t0 = time.time()
        for _ in range(a.bench):
            s.run(None, {'frames': fr.numpy(), 'masks': mk.numpy()})
        print('ORT CPU 4 threads: %.2fs / call (T=%d)' % ((time.time() - t0) / a.bench, T))
    if a.fp16:
        from onnxconverter_common import float16
        onnx.save(float16.convert_float_to_float16(onnx.load(a.out), keep_io_types=True), a.out)
        o16 = ort.InferenceSession(a.out, so, providers=['CPUExecutionProvider']).run(None, {'frames': fr.numpy(), 'masks': mk.numpy()})[0]
        hole = np.broadcast_to(mk.numpy() > 0, o.shape)
        mse = ((o16 - o)[hole] ** 2).mean()
        print('rewrote %s as fp16 weights (fp32 io)  PSNR in hole vs fp32 %.1f dB' % (a.out, 10 * np.log10(1 / max(mse, 1e-12))))


if __name__ == '__main__':
    main()
