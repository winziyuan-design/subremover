# Flow-guided real-pixel propagation ("ProPainter-lite"): DIS flow at half res, flow completed inside holes,
# forward/backward consistency, bidirectional recurrent propagation inside a shot. Returns filled frames + never-seen mask.
import cv2, numpy as np
_dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
S=0.5
def _complete(flow, hole_s):
    """fill flow inside (half-res) hole by inpainting each channel at quarter res, keep outside as is"""
    if not hole_s.any(): return flow
    sm=cv2.resize(flow,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA); hm=cv2.resize(hole_s,(sm.shape[1],sm.shape[0]),interpolation=cv2.INTER_NEAREST)
    hm=cv2.dilate(hm,np.ones((5,5),np.uint8))
    out=np.empty_like(sm)
    for c in range(2):
        ch=sm[...,c]; lo,hi=float(ch.min()),float(ch.max()); sc=255.0/max(hi-lo,1e-3)
        u8=np.clip((ch-lo)*sc,0,255).astype(np.uint8)
        f=cv2.inpaint(u8,hm,5,cv2.INPAINT_TELEA).astype(np.float32)/sc+lo
        out[...,c]=np.where(hm>0,f,ch)
    up=cv2.resize(out,(flow.shape[1],flow.shape[0]),interpolation=cv2.INTER_LINEAR)
    hd=cv2.dilate(hole_s,np.ones((5,5),np.uint8))>0
    flow=flow.copy(); flow[hd]=up[hd]; return flow
def _flow(a,b,ha,hb):
    """completed flows a->b and b->a at half res (pixels), from gray half-res images and half-res holes"""
    fab=_dis.calc(a,b,None); fba=_dis.calc(b,a,None)
    return _complete(fab,ha),_complete(fba,hb)
def propagate(frames, holes, thr=1.0, srcbad=None):
    """frames: list of BGR full-res (one shot, consecutive), holes: list of uint8 masks (255=hole).
    returns filled frames, unseen masks (255=never seen)"""
    n=len(frames); H,W=holes[0].shape
    gs=[cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),None,fx=S,fy=S,interpolation=cv2.INTER_AREA) for f in frames]
    hs=[cv2.resize(h,None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST) for h in holes]
    fw={};bw={}
    for t in range(n-1):
        f01,f10=_flow(gs[t],gs[t+1],hs[t],hs[t+1]); fw[t]=f01; bw[t+1]=f10   # fw[t]: t->t+1 ; bw[t+1]: t+1->t
    gx,gy=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
    def up(fl): return cv2.resize(fl,(W,H),interpolation=cv2.INTER_LINEAR)/S
    res={}
    for direction in (+1,-1):
        img=[f.astype(np.float32) for f in frames]; known=[(h==0)&(True if srcbad is None else srcbad[k]==0) for k,h in enumerate(holes)]; dist=[np.where(h==0,0,999).astype(np.int16) for h in holes]
        order=range(1,n) if direction>0 else range(n-2,-1,-1)
        for t in order:
            s=t-direction                                  # already processed source frame
            ft=up(bw[t] if direction>0 else fw[t])         # t -> s
            fs=up(fw[s] if direction>0 else bw[s])         # s -> t
            mx,my=gx+ft[...,0],gy+ft[...,1]
            back=cv2.remap(fs,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=1e4)
            err=np.hypot(ft[...,0]+back[...,0],ft[...,1]+back[...,1])
            ok_src=cv2.remap(known[s].astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
            ok=(holes[t]>0)&(~known[t])&ok_src&(err<thr+0.05*np.hypot(ft[...,0],ft[...,1]))&(mx>=0)&(my>=0)&(mx<=W-1)&(my<=H-1)
            if ok.any():
                val=cv2.remap(img[s],mx,my,cv2.INTER_LINEAR)
                d=cv2.remap(dist[s].astype(np.float32),mx,my,cv2.INTER_NEAREST)+1
                img[t][ok]=val[ok]; known[t]=known[t]|ok; dist[t][ok]=d[ok].astype(np.int16)
        res[direction]=(img,known,dist)
    out=[];unseen=[]
    for t in range(n):
        (a,ka,da),(b,kb,db)=res[1],res[-1]
        o=frames[t].astype(np.float32); h=holes[t]>0
        fa=ka[t]&h; fb=kb[t]&h
        useA=fa&(~fb|(da[t]<=db[t])); useB=fb&~useA
        both=fa&fb&(np.abs(da[t].astype(int)-db[t])<=2)
        o[useA]=a[t][useA]; o[useB]=b[t][useB]
        o[both&h]=(0.5*(a[t]+b[t]))[both&h]
        out.append(np.clip(o,0,255).astype(np.uint8)); unseen.append(((h&~(fa|fb))*255).astype(np.uint8))
    return out,unseen
