import sys, torch, torch.nn as nn
sys.path.insert(0, '/workspace/vinp/STTN')
import types; sys.modules['torchvision']=types.ModuleType('torchvision'); tm=types.ModuleType('torchvision.models'); sys.modules['torchvision.models']=tm; sys.modules['torchvision'].models=tm
from model.sttn import InpaintGenerator
T = int(sys.argv[1]) if len(sys.argv)>1 else 16
g = InpaintGenerator(init_weights=False)
sd = torch.load('/workspace/vinp/vsr_auto.pth', map_location='cpu')
print(sd.keys() if isinstance(sd, dict) else type(sd))
g.load_state_dict(sd['netG']); g.eval()
print('params %.1fM' % (sum(p.numel() for p in g.parameters())/1e6))
class W(nn.Module):
    def __init__(s, g): super().__init__(); s.g = g
    def forward(s, frames, masks):   # frames [T,3,240,432] in [0,1] RGB, masks [T,1,240,432] 1=hole
        x = (frames*2-1)*(1-masks)
        f = s.g.encoder(x)
        m = torch.nn.functional.interpolate(masks, scale_factor=0.25)
        f = s.g.transformer({'x': f, 'm': m, 'b': 1, 'c': 256})['x']
        o = torch.tanh(s.g.decoder(f))
        return (o+1)/2
w = W(g)
fr = torch.rand(T,3,240,432); mk = torch.zeros(T,1,240,432); mk[:,:,150:200,100:300]=1
for p in w.parameters(): p.requires_grad_(False)
with torch.no_grad(): ref = w(fr, mk)
with torch.no_grad(): torch.onnx.export(w, (fr, mk), f'vsrauto_t{T}.onnx', input_names=['frames','masks'], output_names=['out'], opset_version=17, dynamo=False, do_constant_folding=False)
import onnxruntime as ort, numpy as np, time
so = ort.SessionOptions(); so.intra_op_num_threads = 4
s = ort.InferenceSession(f'vsrauto_t{T}.onnx', so)
import resource; print('rss before run MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
o = s.run(None, {'frames': fr.numpy(), 'masks': mk.numpy()})[0]
print('maxdiff', np.abs(o-ref.numpy()).max())
#t=time.time(); [s.run(None, {'frames': fr.numpy(), 'masks': mk.numpy()}) for _ in range(3)]; print('ort 4thr %.2fs / call (T=%d)' % ((time.time()-t)/3, T))
