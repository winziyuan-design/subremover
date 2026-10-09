# STTN generator loader + the inference wrapper used for ONNX export (frames [T,3,240,432] RGB 0..1, masks [T,1,240,432] 1=hole -> out [T,3,240,432]).
import os, sys, types, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
STTN_REPO=os.environ.get('STTN_REPO',os.path.join(HERE,'..','third_party','STTN'))
STTN_CKPT=os.environ.get('STTN_CKPT',os.path.join(HERE,'..','third_party','sttn.pth'))
TW,TH=432,240
def _import(repo):
    repo=os.path.abspath(repo)
    if repo not in sys.path: sys.path.insert(0,repo)
    try: import torchvision.models  # noqa
    except ImportError:
        tv=types.ModuleType('torchvision'); tm=types.ModuleType('torchvision.models'); tv.models=tm; sys.modules['torchvision']=tv; sys.modules['torchvision.models']=tm
    from model import sttn
    return sttn
def generator(repo=None, ckpt=None, init=False):
    g=_import(repo or STTN_REPO).InpaintGenerator(init_weights=init)
    ck=ckpt if ckpt is not None else STTN_CKPT
    if ck:
        sd=torch.load(ck,map_location='cpu'); g.load_state_dict(sd.get('netG',sd))
    return g
def discriminator(repo=None):
    return _import(repo or STTN_REPO).Discriminator(in_channels=3,use_sigmoid=False)
class Wrap(nn.Module):
    def __init__(s, g): super().__init__(); s.g=g
    def forward(s, frames, masks):
        x=(frames*2-1)*(1-masks)
        f=s.g.encoder(x)
        m=F.interpolate(masks,scale_factor=0.25)
        f=s.g.transformer({'x':f,'m':m,'b':1,'c':256})['x']
        return (torch.tanh(s.g.decoder(f))+1)/2
def runner(g=None, onnx=None, threads=4):
    """callable(fr[T,3,240,432] f32, mk[T,1,240,432] f32) -> out[T,3,240,432] f32 numpy"""
    if onnx:
        import onnxruntime as ort
        so=ort.SessionOptions(); so.intra_op_num_threads=threads; so.enable_cpu_mem_arena=False
        s=ort.InferenceSession(onnx,so,providers=['CPUExecutionProvider'])
        return lambda fr,mk: s.run(None,{'frames':fr,'masks':mk})[0]
    torch.set_num_threads(threads); w=Wrap(g).eval()
    def run(fr,mk):
        with torch.inference_mode(): return w(torch.from_numpy(fr),torch.from_numpy(mk)).numpy()
    return run
