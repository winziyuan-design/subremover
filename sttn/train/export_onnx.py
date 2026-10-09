# Fine-tuned netG -> ONNX with the same interface as ../export.py: frames [T,3,240,432] RGB 0..1, masks [T,1,240,432] (1=hole) -> out [T,3,240,432].
import os, sys, argparse, numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sttn_net
ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',default=sttn_net.STTN_CKPT); ap.add_argument('--sttn-repo',default=sttn_net.STTN_REPO)
ap.add_argument('--T',type=int,default=16); ap.add_argument('--out'); ap.add_argument('--fp16',action='store_true'); a=ap.parse_args()
out=a.out or f'sttn_ft_t{a.T}.onnx'
g=sttn_net.generator(a.sttn_repo,a.ckpt).eval(); w=sttn_net.Wrap(g)
for p in w.parameters(): p.requires_grad_(False)
fr=torch.rand(a.T,3,240,432); mk=torch.zeros(a.T,1,240,432); mk[:,:,150:200,100:300]=1
with torch.no_grad():
    ref=w(fr,mk).numpy()
    torch.onnx.export(w,(fr,mk),out,input_names=['frames','masks'],output_names=['out'],opset_version=17,dynamo=False,do_constant_folding=False)
if a.fp16:
    import onnx
    from onnxconverter_common import float16
    onnx.save(float16.convert_float_to_float16(onnx.load(out),keep_io_types=True),out)
import onnxruntime as ort
o=ort.InferenceSession(out,providers=['CPUExecutionProvider']).run(None,{'frames':fr.numpy(),'masks':mk.numpy()})[0]
print(out,'%.1f MB'%(os.path.getsize(out)/2**20),'maxdiff %.2e'%np.abs(o-ref).max())
