import sys, torch, types, time
sys.path.insert(0, '/workspace/vinp/STTN')
sys.modules['torchvision']=types.ModuleType('torchvision'); tm=types.ModuleType('torchvision.models'); sys.modules['torchvision.models']=tm
from model.sttn import InpaintGenerator
g = InpaintGenerator(init_weights=False); g.load_state_dict(torch.load('/workspace/vinp/sttn.pth', map_location='cpu')['netG']); g.eval()
T=int(sys.argv[1]); torch.set_num_threads(4)
x=torch.rand(1,T,3,240,432); m=torch.zeros(1,T,1,240,432)
with torch.no_grad():
    t=time.time(); g(x,m); print('torch T=%d %.2fs'%(T,time.time()-t))
