# STTN on full-res regions: (B) native-resolution 432x240 tiles with overlap + feathered blend, and (A) the legacy pipe.py layout (region resized onto one 432x240 canvas).
# frames: list of T BGR uint8 (same size), holes: list of T uint8 (255=hole), run: sttn_net.runner(). Returns T BGR frames, changed only in/around holes.
import cv2, numpy as np
TW,TH=432,240
def _view(img,rot): return cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) if rot else img
def _unview(img,rot): return cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) if rot else img
def _bbox(holes):
    U=np.zeros_like(holes[0])
    for h in holes: U|=h
    ys,xs=np.nonzero(U)
    return None if not len(xs) else (xs.min(),ys.min(),xs.max()+1,ys.max()+1)
def _feather(h, k=9):
    m=(h>0).astype(np.uint8)
    soft=cv2.GaussianBlur(cv2.dilate(m,np.ones((k,k),np.uint8)).astype(np.float32),(k,k),0)
    return np.maximum(soft,m.astype(np.float32))
def _batch(crops, hcrops):
    fr=(np.stack(crops).astype(np.float32)/255)[...,::-1].transpose(0,3,1,2).copy()
    mk=(np.stack(hcrops)>0).astype(np.float32)[:,None]
    return fr,mk
def _unbatch(o): return np.clip(o.transpose(0,2,3,1)[...,::-1],0,1)*255
def _starts(lo, hi, n, tile, ov):
    """tile origins covering [lo,hi) inside [0,n) with >= ov overlap"""
    span=hi-lo
    if span<=tile:
        return [int(np.clip(lo+span//2-tile//2,0,n-tile))]
    k=int(np.ceil((span-ov)/(tile-ov)))
    return [int(np.clip(round(lo+i*(span-tile)/(k-1)),0,n-tile)) for i in range(k)] if k>1 else [lo]
def _ramp(n, tile, s, ov):
    """1D blend weight for a tile at origin s: linear ramps over ov at edges that are not image borders"""
    w=np.ones(tile,np.float32); r=np.linspace(1.0/(ov+1),1.0,ov,dtype=np.float32) if ov>0 else None
    if r is not None:
        if s>0: w[:ov]=np.minimum(w[:ov],r)
        if s+tile<n: w[-ov:]=np.minimum(w[-ov:],r[::-1])
    return w
def plan(holes, ov=96, ctx=64, rot='auto', scale=1.0):
    """-> dict(rot, scale, tiles=[(x,y)] in the scaled view) or None if no hole"""
    bb=_bbox(holes)
    if bb is None: return None
    H,W=holes[0].shape; x0,y0,x1,y1=bb
    r=(y1-y0)>(x1-x0)*1.2 if rot=='auto' else bool(rot)
    if r: x0,y0,x1,y1=y0,W-x1,y1,W-x0; W,H=H,W
    s=scale
    if s=='auto': s=min(1.0,(TH-2*ctx)/max(1,y1-y0)) if (y1-y0)>TH-2*ctx else 1.0
    s=float(min(s,1.0)); W2,H2=max(TW,int(round(W*s))),max(TH,int(round(H*s)))
    bx0,by0,bx1,by1=[int(v*s) for v in (x0,y0,x1,y1)]
    bx0,by0,bx1,by1=max(0,bx0-ctx),max(0,by0-ctx),min(W2,bx1+ctx),min(H2,by1+ctx)
    tiles=[(x,y) for y in _starts(by0,by1,H2,TH,ov) for x in _starts(bx0,bx1,W2,TW,ov)]
    return dict(rot=r,scale=s,size=(W2,H2),tiles=tiles,ov=ov)
def inpaint_tiles(frames, holes, run, ov=96, ctx=64, rot='auto', scale=1.0, feather=9, info=None):
    p=plan(holes,ov,ctx,rot,scale)
    if p is None: return [f.copy() for f in frames]
    r,s,(W2,H2)=p['rot'],p['scale'],p['size']
    vf=[_view(f,r) for f in frames]; vh=[_view(h,r) for h in holes]; Hv,Wv=vh[0].shape
    if s!=1.0 or (W2,H2)!=(Wv,Hv):
        sf=[cv2.resize(f,(W2,H2),interpolation=cv2.INTER_AREA) for f in vf]
        sh=[(cv2.resize(h,(W2,H2),interpolation=cv2.INTER_AREA)>0).astype(np.uint8)*255 for h in vh]
    else: sf,sh=vf,vh
    T=len(frames); num=np.zeros((T,H2,W2,3),np.float32); den=np.zeros((T,H2,W2),np.float32); calls=0
    for (x,y) in p['tiles']:
        hc=[cv2.dilate(h[y:y+TH,x:x+TW],np.ones((3,3),np.uint8)) for h in sh]
        if not any(c.any() for c in hc): continue
        o=_unbatch(run(*_batch([f[y:y+TH,x:x+TW] for f in sf],hc))); calls+=1
        w=np.outer(_ramp(H2,TH,y,ov),_ramp(W2,TW,x,ov))
        num[:,y:y+TH,x:x+TW]+=o*w[None,...,None]; den[:,y:y+TH,x:x+TW]+=w[None]
    if info is not None: info.update(calls=calls,tiles=len(p['tiles']),scale=s,rot=r)
    out=[]
    for t in range(T):
        fill=num[t]/np.maximum(den[t],1e-6)[...,None]
        if (W2,H2)!=(Wv,Hv): fill=cv2.resize(fill,(Wv,Hv),interpolation=cv2.INTER_CUBIC); cov=cv2.resize(den[t],(Wv,Hv))>1e-6
        else: cov=den[t]>1e-6
        a=_feather(vh[t],feather)*cov; F=vf[t].astype(np.float32)
        out.append(_unview(np.clip(fill*a[...,None]+F*(1-a[...,None])+0.5,0,255).astype(np.uint8),r))
    return out
def _layout(rect, W, H):
    x0,y0,x1,y1=rect; w,h=x1-x0,y1-y0; rot=h>w*1.2
    if rot: w,h=h,w
    k=max(1,int(round(np.sqrt(w/h/1.8)))); asp=TW/(TH/k); pieces=[]
    for j in range(k):
        cx0=(y0 if rot else x0)+j*w//k; cx1=(y0 if rot else x0)+(j+1)*w//k; vy0=(W-x1) if rot else y0; vy1=(W-x0) if rot else y1
        pw,ph=cx1-cx0,vy1-vy0
        if pw/ph>asp: Wc,Hc=pw,int(np.ceil(pw/asp))
        else: Wc,Hc=int(np.ceil(ph*asp)),ph
        VW,VH=(H,W) if rot else (W,H); Wc,Hc=min(Wc,VW),min(Hc,VH)
        bx=int(np.clip((cx0+cx1)//2-Wc//2,0,VW-Wc)); by=int(np.clip((vy0+vy1)//2-Hc//2,0,VH-Hc))
        pieces.append(((cx0,vy0,cx1,vy1),(bx,by,bx+Wc,by+Hc)))
    return rot,k,pieces
def _canvas(img, rot, k, pieces):
    v=_view(img,rot); rh=TH//k; rows=[]
    for core,(bx,by,bx2,by2) in pieces:
        rows.append(cv2.resize(v[by:by2,bx:bx2],(TW,rh),interpolation=cv2.INTER_AREA if img.ndim==3 else cv2.INTER_NEAREST))
    c=np.vstack(rows)
    return c if c.shape[0]==TH else np.concatenate([c,np.repeat(c[-1:],TH-c.shape[0],0)],0)
def inpaint_legacy(frames, holes, run, mg=12, feather=9, info=None):
    bb=_bbox(holes)
    if bb is None: return [f.copy() for f in frames]
    H,W=holes[0].shape; x0,y0,x1,y1=bb
    rot,k,pieces=_layout((max(0,x0-mg),max(0,y0-mg),min(W,x1+mg),min(H,y1+mg)),W,H)
    fr,mk=_batch([_canvas(f,rot,k,pieces) for f in frames],[cv2.dilate(_canvas(h,rot,k,pieces),np.ones((3,3),np.uint8)) for h in holes])
    o=_unbatch(run(fr,mk)).astype(np.uint8); rh=TH//k
    if info is not None: info.update(calls=1,k=k,rot=rot,scale=round(TW/(pieces[0][1][2]-pieces[0][1][0]),3))
    out=[]
    for t in range(len(frames)):
        vF=_view(frames[t],rot).copy(); vH=_view(holes[t],rot)
        for j,((cx0,cy0,cx1,cy1),(bx,by,bx2,by2)) in enumerate(pieces):
            tile=cv2.resize(o[t][j*rh:(j+1)*rh],(bx2-bx,by2-by),interpolation=cv2.INTER_CUBIC)
            sl=(slice(by,by2),slice(max(bx,cx0 if k>1 else bx),min(bx2,cx1 if k>1 else bx2)))
            a=_feather(vH[sl],feather)[...,None]; tl=tile[:,sl[1].start-bx:sl[1].stop-bx]
            vF[sl]=(tl*a+vF[sl]*(1-a)).astype(np.uint8)
        out.append(_unview(vF,rot))
    return out
