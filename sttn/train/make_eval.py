# Held-out synthetic eval clips (fixed seed, inference shape T=16 = 10 local + 6 refs): <out>/NNNN.npz with gt/inp BGR uint8 [T,240,432,3], mask uint8 [T,240,432].
import os, sys, argparse, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import dataset
ap=argparse.ArgumentParser(); ap.add_argument('--data',nargs='+',required=True); ap.add_argument('--out',required=True); ap.add_argument('--n',type=int,default=200)
ap.add_argument('--T',type=int,default=16); ap.add_argument('--nb',type=int,default=10); ap.add_argument('--seed',type=int,default=1234); ap.add_argument('--fonts',nargs='*'); ap.add_argument('--corpus')
ap.add_argument('--p-free',type=float,default=0.0); ap.add_argument('--scale',type=float,nargs=2,default=[1.0,1.0]); a=ap.parse_args()
ds=dataset.Clips(a.data,T=a.T,nb=a.nb,scale=tuple(a.scale),p_rot=0.0,p_free=a.p_free,fonts=a.fonts,corpus=a.corpus,seed=a.seed)
os.makedirs(a.out,exist_ok=True)
u8=lambda x: np.clip((x.permute(0,2,3,1).numpy()[...,::-1]+1)*127.5+0.5,0,255).astype(np.uint8)
for i in range(a.n):
    gt,inp,m=ds[i]
    np.savez_compressed(os.path.join(a.out,f'{i:04d}.npz'),gt=u8(gt),inp=u8(inp),mask=(m[:,0].numpy()*255).astype(np.uint8))
print('wrote',a.n,'clips ->',a.out)
