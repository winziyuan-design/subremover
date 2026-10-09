# Training samples: T frames from a clean clip, 432x240 crop at native scale (optionally slight downscale / 90-degree rotation), synthetic subtitles/watermarks burned in.
# Returns gt [T,3,H,W] in -1..1 (RGB), inp [T,3,H,W] (overlaid, maybe JPEG'd), mask [T,1,H,W] (1=hole).
import os, glob, numpy as np, cv2, torch
import synth
def list_clips(roots, min_len=8):
    out=[]
    for r in roots:
        for d in sorted(glob.glob(os.path.join(r,'*'))):
            fs=sorted(glob.glob(os.path.join(d,'*.jpg'))+glob.glob(os.path.join(d,'*.png')))
            if len(fs)>=min_len: out.append(fs)
    if not out: raise RuntimeError(f'no clips under {roots} (run prep_frames.py)')
    return out
def free_mask(rng, T, w, h):
    """STTN-style random blob, static or drifting"""
    pts=int(rng.integers(6,10)); cx,cy=rng.uniform(0.2,0.8)*w,rng.uniform(0.2,0.8)*h; R=rng.uniform(0.1,0.35)*h
    ang=np.sort(rng.uniform(0,2*np.pi,pts)); rad=R*rng.uniform(0.5,1.3,pts)
    poly=np.stack([np.cos(ang)*rad,np.sin(ang)*rad],1); v=rng.uniform(-3,3,2) if rng.random()<0.5 else np.zeros(2)
    ms=[]
    for t in range(T):
        m=np.zeros((h,w),np.uint8); cv2.fillPoly(m,[(poly+[cx+v[0]*t,cy+v[1]*t]).astype(np.int32)],255); ms.append(cv2.GaussianBlur(m,(7,7),0)>64)
    return ms
class Clips(torch.utils.data.Dataset):
    def __init__(s, roots, T=5, nb=3, w=432, h=240, scale=(0.7,1.0), p_rot=0.15, p_free=0.15, p_jpeg=0.7, fonts=None, corpus=None, length=10**6, seed=None):
        s.clips=list_clips(roots); s.T,s.nb,s.w,s.h,s.scale,s.p_rot,s.p_free,s.p_jpeg,s.length,s.seed=T,min(nb,T),w,h,scale,p_rot,p_free,p_jpeg,length,seed
        s.fonts=synth.find_fonts(fonts); s.corpus=[l for l in open(corpus,encoding='utf-8')] if corpus else None
    def __len__(s): return s.length
    def _rng(s, i): return np.random.default_rng(None if s.seed is None else (s.seed,i))
    def _ids(s, rng, L):
        st=int(rng.integers(1,4)); span=min(L,s.nb*st); p=int(rng.integers(0,L-span+1)); loc=list(range(p,p+span,st))[:s.nb]
        rest=[i for i in range(L) if i not in loc]; nr=s.T-len(loc)
        ref=sorted(rng.choice(rest,nr,replace=len(rest)<nr).tolist()) if nr>0 else []
        return loc+ref if rng.random()<0.5 else sorted(loc+ref)
    def __getitem__(s, i):
        rng=s._rng(i); fs=s.clips[int(rng.integers(len(s.clips)))]; ids=s._ids(rng,len(fs))
        rot=rng.random()<s.p_rot; sc=rng.uniform(*s.scale); W,H=(s.h,s.w) if rot else (s.w,s.h)
        cw,ch=int(round(W/sc)),int(round(H/sc)); fr=[cv2.imread(fs[j],cv2.IMREAD_COLOR) for j in ids]
        fh,fw=fr[0].shape[:2]
        if fw<cw or fh<ch:
            k=max(cw/fw,ch/fh); fr=[cv2.resize(f,None,fx=k,fy=k,interpolation=cv2.INTER_CUBIC) for f in fr]; fh,fw=fr[0].shape[:2]
        x,y=int(rng.integers(0,fw-cw+1)),int(rng.integers(0,fh-ch+1))
        fr=[f[y:y+ch,x:x+cw] for f in fr]
        if (cw,ch)!=(W,H): fr=[cv2.resize(f,(W,H),interpolation=cv2.INTER_AREA) for f in fr]
        if rng.random()<0.5: fr=[f[:,::-1] for f in fr]
        span=max(ids)-min(ids)+1; t0=min(ids)
        ov=synth.Overlay(W,H,span,rng,s.fonts,mode='crop',corpus=s.corpus,p_gap=0.05,seg=(max(2,span//4),max(3,span)))
        fm=free_mask(rng,s.T,s.w,s.h) if rng.random()<s.p_free else None
        dil=int(rng.integers(1,6)); q=int(rng.integers(50,96)) if rng.random()<s.p_jpeg else 0
        R=(lambda a: cv2.rotate(a,cv2.ROTATE_90_COUNTERCLOCKWISE)) if rot else (lambda a: a)
        gt=[];inp=[];mk=[]
        for k,(j,f) in enumerate(zip(ids,fr)):
            f=np.ascontiguousarray(f); o,m=ov.apply(f,j-t0,dil=dil); f,o,m=R(f),R(o),R(m)
            if fm is not None: m=m|(fm[k].astype(np.uint8)*255)
            if q: o=synth.jpeg(o,q)
            gt.append(f); inp.append(o); mk.append(m)
        cv=lambda a: torch.from_numpy(np.stack(a)[...,::-1].astype(np.float32)/127.5-1).permute(0,3,1,2).contiguous()
        return cv(gt),cv(inp),torch.from_numpy((np.stack(mk)>0).astype(np.float32))[:,None]
