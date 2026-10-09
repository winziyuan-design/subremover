# Clean videos -> shot-split clips of JPEG frames (<root>/<video>_<k>/00000.jpg ...) at native resolution (short side capped). Videos must be free of burned-in text.
import os, sys, glob, argparse, cv2, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument('inputs',nargs='+'); ap.add_argument('--out',required=True)
ap.add_argument('--max-short',type=int,default=1080); ap.add_argument('--min-len',type=int,default=24); ap.add_argument('--max-len',type=int,default=150)
ap.add_argument('--stride',type=int,default=1); ap.add_argument('--cut',type=float,default=25.0); ap.add_argument('--quality',type=int,default=95)
ap.add_argument('--skip-start',type=float,default=0.0); ap.add_argument('--min-std',type=float,default=12.0); a=ap.parse_args()
vids=[f for p in a.inputs for f in (sorted(glob.glob(os.path.join(p,'**','*.*'),recursive=True)) if os.path.isdir(p) else [p]) if f.lower().endswith(('.mp4','.mkv','.webm','.mov','.avi','.m4v'))]
os.makedirs(a.out,exist_ok=True); total=0
for v in vids:
    cap=cv2.VideoCapture(v); fps=cap.get(5) or 30; n=0; k=0; cur=[]; prev=None; name=os.path.splitext(os.path.basename(v))[0]
    def flush():
        global k,total
        if len(cur)>=a.min_len:
            d=os.path.join(a.out,f'{name}_{k:04d}'); os.makedirs(d,exist_ok=True)
            for j,f in enumerate(cur): cv2.imwrite(os.path.join(d,f'{j:05d}.jpg'),f,[cv2.IMWRITE_JPEG_QUALITY,a.quality])
            k+=1; total+=len(cur)
        cur.clear()
    while True:
        ok,f=cap.read()
        if not ok: break
        n+=1
        if n<a.skip_start*fps or (n-1)%a.stride: continue
        s=a.max_short/min(f.shape[:2])
        if s<1: f=cv2.resize(f,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
        tiny=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(64,36)).astype(np.float32)
        if prev is not None and np.abs(tiny-prev).mean()>a.cut: flush()
        prev=tiny
        if tiny.std()<a.min_std: flush(); continue
        cur.append(f)
        if len(cur)>=a.max_len: flush()
    flush(); print(v,'clips',k,flush=True)
print('frames',total,'->',a.out)
