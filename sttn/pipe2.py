# Box prototype: subtitle strokes + static watermark + moving watermark masks -> STTN (ONNX) on region crops -> feathered paste-back.
import cv2, numpy as np, sys, time, json, onnxruntime as ort, argparse
ap=argparse.ArgumentParser(); ap.add_argument('--model',default='sttn_t16.onnx'); ap.add_argument('--frames',default='all')
ap.add_argument('--out',default='out_sttn'); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--prop',type=int,default=1); ap.add_argument('--R',type=int,default=20); ap.add_argument('--nb',type=int,default=10); ap.add_argument('--sreuse',type=int,default=0); ap.add_argument('--tiny',type=int,default=0); a_=ap.parse_args()
SRC='/workspace/subremover/sample/a.mp4'; DET='/workspace/subremover/third_party/det.onnx'
T0=time.time(); tm={}
# ---------- decode ----------
cap=cv2.VideoCapture(SRC); FPS=cap.get(5); frames=[]
while True:
    ok,f=cap.read()
    if not ok: break
    frames.append(cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,97])[1])
N=len(frames); H,W=1920,1080
def FR(i): return cv2.imdecode(frames[i],cv2.IMREAD_COLOR)
tm['decode']=time.time()-T0
# ---------- subtitle stroke masks (same idea as v0.2: DB boxes in a stable band + bright low-sat strokes stable over time) ----------
t=time.time()
det=cv2.dnn_TextDetectionModel_DB(DET); det.setBinaryThreshold(0.3); det.setPolygonThreshold(0.5); det.setMaxCandidates(200); det.setUnclipRatio(2.0)
MEAN=(122.67891434,116.66876762,104.00698793)
def detect(img,maxw):
    h,w=img.shape[:2]; s=min(1.0,maxw/w); iw=max(32,int(round(w*s/32))*32); ih=max(32,int(round(h*s/32))*32)
    det.setInputParams(1.0/255,(iw,ih),MEAN); return [cv2.boundingRect(np.array(p,np.int32)) for p in det.detect(img)[0]]
cand=[]
for i in np.linspace(0,N-1,10).astype(int):
    top=H*2//5
    for (x,y,bw,bh) in detect(FR(int(i))[top:],960):
        y+=top; cx=x+bw/2
        if y+bh/2>H*0.5 and abs(cx-W/2)<W*0.12 and H*0.02<bh<H*0.14 and bw>bh*1.5: cand.append((y,y+bh))
cy=np.median([(p+q)/2 for p,q in cand]); LH=np.median([q-p for p,q in cand])
keep=[(p,q) for p,q in cand if abs((p+q)/2-cy)<LH*1.6]
y0=max(0,int(min(p for p,q in keep)-LH*0.6))&~1; y1=min(H,int(max(q for p,q in keep)+LH*0.6))
bands=[FR(i)[y0:y1].copy() for i in range(N)]; bh_,bw_=bands[0].shape[:2]
hatK=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(LH*0.5)|1,int(LH*0.5)|1))
K=3
def _centered_run(cols, gap):
    """contiguous run of active columns around the frame centre (gaps < gap bridged)"""
    act=np.nonzero(cols)[0]
    if not len(act): return None
    c=W//2; j=np.argmin(np.abs(act-c))
    if abs(act[j]-c)>W*0.15: return None
    lo=hi=j
    while lo>0 and act[lo]-act[lo-1]<gap: lo-=1
    while hi<len(act)-1 and act[hi+1]-act[hi]<gap: hi+=1
    return act[lo],act[hi]
def line_boxes(i):
    """centred subtitle line in band frame i: DB boxes chained along the line, else a bright-stroke run around the centre"""
    rs=[r for r in detect(bands[i],640) if LH*0.35<r[3]<LH*1.8]
    if rs:
        rs.sort(key=lambda r:abs(r[0]+r[2]/2-W/2)); r0=rs[0]; grp=[r0]; ch=True
        while ch:
            ch=False
            for r in rs:
                if r in grp: continue
                gx0=min(g[0] for g in grp); gx1=max(g[0]+g[2] for g in grp)
                if (r[0]<gx1+LH*0.4 and r[0]+r[2]>gx0-LH*0.4 and abs((r[1]+r[3]/2)-(r0[1]+r0[3]/2))<LH*0.3
                        and abs(r[3]-r0[3])<0.35*r0[3]): grp.append(r); ch=True
        for cand in (grp,[r0]):
            gx0=min(g[0] for g in cand); gx1=max(g[0]+g[2] for g in cand)
            if abs((gx0+gx1)/2-W/2)<W*0.03: return cand
    # short lines (2-3 chars) on bright backgrounds are missed at 640px: re-detect a centre crop upscaled 1.5x
    cx0=W//2-240; crop=bands[i][:,cx0:cx0+480]; h,w=crop.shape[:2]
    det.setInputParams(1.0/255,(int(w*1.5)//32*32,int(h*1.5)//32*32+32),MEAN)
    for p in det.detect(crop)[0]:
        x,y,w_,h_=cv2.boundingRect(np.array(p,np.int32)); x+=cx0
        if abs(x+w_/2-W/2)<W*0.03 and LH*0.5<h_<LH*1.8 and w_>LH*0.8: return [(x,max(0,y),w_,h_)]
    return []
dets={i:line_boxes(i) for i in range(0,N,K)}
pad=int(LH*0.25)+2
dk=max(3,int(round(LH*0.2))); dilK=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dk+1,2*dk+1))
def region(i, reach=1):
    b=i//K*K; m=np.zeros((bh_,bw_),np.uint8)
    for j in range(b-reach*K,b+(reach+1)*K,K):
        for (x,y,w,h) in dets.get(j,[]): cv2.rectangle(m,(int(x-pad),int(y-pad)),(int(x+w+pad),int(y+h+pad)),255,-1)
    return m
_raw={}
def raw(i):
    i=min(max(i,0),N-1)
    if i not in _raw:
        reg=region(i)
        if not reg.any(): _raw[i]=reg
        else:
            hsv=cv2.cvtColor(bands[i],cv2.COLOR_BGR2HSV); g=hsv[...,2]; th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,hatK)
            _raw[i]=((th>35)&(g>170)&(hsv[...,1]<80)).astype(np.uint8)*255&reg
    return _raw[i]
def stroke(i):
    s=(raw(i)&raw(i-2))|(raw(i)&raw(i+2)); s=cv2.morphologyEx(s,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return cv2.dilate(s,dilK)&cv2.dilate(region(i),dilK)
subm=[stroke(i) for i in range(N)]
tm['sub_mask']=time.time()-t; print('band',y0,y1,'LH',LH)
# ---------- static watermark: edges / bright strokes that persist over sampled frames ----------
t=time.time(); ed=[]; br=[]
for i in np.linspace(0,N-1,60).astype(int):
    g=cv2.cvtColor(FR(int(i)),cv2.COLOR_BGR2GRAY); ed.append(cv2.Canny(g,60,160)>0); br.append(cv2.morphologyEx(g,cv2.MORPH_TOPHAT,np.ones((15,15),np.uint8)))
st=((np.mean(ed,0)>0.5)|(np.median(br,0)>60)).astype(np.uint8)*255
st[max(0,y0-40):y1+40]=0
n,lab,ss,_=cv2.connectedComponentsWithStats(cv2.dilate(st,np.ones((15,15),np.uint8)))
statm=np.zeros((H,W),np.uint8)
for k in range(1,n):
    if ss[k,4]>400: statm[lab==k]=255
statm=cv2.dilate(st,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))&statm
tm['static_mask']=time.time()-t
# ---------- moving watermark (template found by DB-box recurrence, tracked by template matching; see moving_wm.py) ----------
mw=np.load('moving_wm.npz'); mpos=np.load('wm_pos_refined.npy').astype(float); mmask=cv2.copyMakeBorder(mw['mask'],16,16,16,16,cv2.BORDER_CONSTANT,value=0); mpos=mpos-16; mh,mwid=mmask.shape
n_,lab_,ss_,_=cv2.connectedComponentsWithStats(mmask); big=ss_[1:,4].max()
for k_ in range(1,n_):
    if ss_[k_,4]<0.1*big: mmask[lab_==k_]=0
def movm(i):
    m=np.zeros((H,W),np.uint8); x,y=int(round(mpos[i,0])),int(round(mpos[i,1]))
    xa,ya=max(0,x),max(0,y); xb,yb=min(W,x+mwid),min(H,y+mh)
    if xb>xa and yb>ya: m[ya:yb,xa:xb]=mmask[ya-y:yb-y,xa-x:xb-x]
    return m
tm['moving_mask']=35.0   # measured separately (moving_wm.py)
# ---------- shot cuts ----------
tiny=[cv2.resize(cv2.cvtColor(FR(i),cv2.COLOR_BGR2GRAY),(36,64)).astype(np.float32) for i in range(N)]
cut=[0]+[1 if np.abs(tiny[i]-tiny[i-1]).mean()>25 else 0 for i in range(1,N)]
shot=np.cumsum(cut); print('shots',shot[-1]+1)
# ---------- regions ----------
def full_sub(i):
    m=np.zeros((H,W),np.uint8); m[y0:y1]=subm[i]; return m
def full_subhole(i):
    m=np.zeros((H,W),np.uint8); m[y0:y1]=cv2.dilate(region(i,2),dilK)|subm[i]; return m
REG={'sub':full_sub,'static':lambda i:statm,'moving':movm}
statH=cv2.dilate(statm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
_mvK=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
HOLE={'sub':full_subhole,'static':lambda i:statH,'moving':lambda i:cv2.dilate(movm(i),_mvK)}
sess_o=ort.SessionOptions(); sess_o.intra_op_num_threads=a_.threads; sess_o.enable_cpu_mem_arena=False
sess=ort.InferenceSession(a_.model,sess_o,providers=['CPUExecutionProvider']); TT=sess.get_inputs()[0].shape[0]
NB=a_.nb; NR=TT-NB; OFF={6:[-90,-60,-30,30,60,90],4:[-75,-40,40,75],3:[-60,45,90],2:[-50,50]}[NR]
MW_,MH_=432,240
def layout(rect):
    """rect (x0,y0,x1,y1) -> list of pieces: (core rect, context rect, rotate) mapped to canvas rows"""
    x0,y0_,x1,y1_=rect; w,h=x1-x0,y1_-y0_; rot=h>w*1.2
    if rot: w,h=h,w
    k=max(1,int(round(np.sqrt(w/h/1.8)))); asp=MW_/(MH_/k)
    pieces=[]
    for j in range(k):
        # in "view" coordinates (rotated: view x = original y)
        cx0=(y0_ if rot else x0)+j*w//k; cx1=(y0_ if rot else x0)+(j+1)*w//k; vy0=(W-x1) if rot else y0_; vy1=(W-x0) if rot else y1_
        pw,ph=cx1-cx0,vy1-vy0
        if pw/ph>asp: Wc,Hc=pw,int(np.ceil(pw/asp))
        else: Wc,Hc=int(np.ceil(ph*asp)),ph
        VW,VH=(H,W) if rot else (W,H)
        Wc,Hc=min(Wc,VW),min(Hc,VH)
        bx=int(np.clip((cx0+cx1)//2-Wc//2,0,VW-Wc)); by=int(np.clip((vy0+vy1)//2-Hc//2,0,VH-Hc))
        pieces.append(((cx0,vy0,cx1,vy1),(bx,by,bx+Wc,by+Hc)))
    return rot,k,pieces
def view(img,rot): return cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) if rot else img
def unview(img,rot): return cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) if rot else img
# view coords for ROTATE_90_COUNTERCLOCKWISE: view(x',y') = orig(x=W-1-y', y=x')  -> orig y range maps to view x directly, orig x maps to view y flipped
def to_view_rect(rect,rot):
    x0,y0_,x1,y1_=rect
    return (y0_,W-x1,y1_,W-x0) if rot else rect
def canvas(img,rot,k,pieces):
    v=view(img,rot); rows=[]
    for core,ctx in pieces:
        bx,by,bx2,by2=ctx; rows.append(cv2.resize(v[by:by2,bx:bx2],(MW_,MH_//k),interpolation=cv2.INTER_AREA if img.ndim==3 else cv2.INTER_NEAREST))
    return np.vstack(rows)
out_full={}   # frame -> composited full frame (lazy)
def get_out(i):
    return cv2.imdecode(out_full[i],cv2.IMREAD_COLOR) if i in out_full else FRP(i)
want=list(range(N)) if a_.frames=='all' else sorted({f for c in a_.frames.split(',') for f in range(int(c.split(':')[0]),int(c.split(':')[1]))})
import prop
_k7=np.ones((7,7),np.uint8); _k15=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
def PU(i): return cv2.dilate(full_sub(i),_k7)|statH|cv2.dilate(movm(i),_mvK)
def PBAD(i):
    hsv=cv2.cvtColor(bands[i],cv2.COLOR_BGR2HSV); g=hsv[...,2]; th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,hatK)
    r=((th>35)&(g>170)&(hsv[...,1]<80)).astype(np.uint8)*255; r[:,:W//2-int(W*0.42)]=0; r[:,W//2+int(W*0.42):]=0
    m=np.zeros((H,W),np.uint8); m[y0:y1]=cv2.dilate(r,_k15); return m|cv2.dilate(PU(i),_k7)
PROP={}   # i -> (jpeg of propagated frame, propagated-pixel mask)
tp=time.time()
if a_.prop:
    # group wanted frames into runs inside a shot, propagate over run +-R
    R=a_.R; runs=[]; cur=[]
    for i in want:
        if cur and (shot[i]!=shot[cur[-1]] or i!=cur[-1]+1): runs.append(cur); cur=[]
        cur.append(i)
    if cur: runs.append(cur)
    for run in runs:
        s_ids=np.nonzero(shot==shot[run[0]])[0]; lo,hi=max(s_ids[0],run[0]-R),min(s_ids[-1],run[-1]+R)
        ids=list(range(lo,hi+1)); holes=[PU(i) for i in ids]
        out,un=prop.propagate([FR(i) for i in ids],holes,srcbad=[PBAD(i) for i in ids])
        for k,i in enumerate(ids):
            if i in run:
                got=((holes[k]>0)&(un[k]==0)).astype(np.uint8)*255
                PROP[i]=(cv2.imencode('.jpg',out[k],[cv2.IMWRITE_JPEG_QUALITY,97])[1],got)
tm['propagate']=time.time()-tp
_FR0=FR
def FRP(i): return cv2.imdecode(PROP[i][0],cv2.IMREAD_COLOR) if i in PROP else _FR0(i)
def GOT(i): return cv2.erode(PROP[i][1],_k7) if i in PROP else None
for _r in list(HOLE):
    def _mk(h):
        def f(i):
            m=h(i); g=GOT(i)
            return m if g is None else (m&~g)
        return f
    HOLE[_r]=_mk(HOLE[_r])
stats={'calls':0,'infer_s':0.0}
SEG={}   # frame -> segment id (static-region reuse)
if a_.sreuse:
    ring=cv2.dilate(statH,np.ones((31,31),np.uint8))&~cv2.dilate(statH,np.ones((7,7),np.uint8)); rr=ring>0
    rv=lambda i: cv2.cvtColor(FR(i),cv2.COLOR_BGR2GRAY)[rr].astype(np.float32)
    sid=0; i=0
    while i<N:
        a=rv(i); j=i+1
        while j<N and shot[j]==shot[i] and np.abs(rv(j)-a).mean()<4.0: j+=1
        if j-i>=8:
            for q in range(i,j): SEG[q]=(sid,i,j)
            sid+=1
        i=j
    stats['static_segments']=sid; stats['static_seg_frames']=len(SEG)

t=time.time()
for rname,mf in REG.items():
    hf=HOLE[rname]
    # chunk wanted frames into NB-runs inside a shot
    chunks=[]; cur=[]
    for i in want:
        if not mf(i).any(): continue
        if cur and (shot[i]!=shot[cur[-1]] or i!=cur[-1]+1 or len(cur)==NB): chunks.append(cur); cur=[]
        cur.append(i)
    if cur: chunks.append(cur)
    reuse={}
    if rname=='static' and SEG:
        segs={}
        for i in want:
            if i in SEG: segs.setdefault(SEG[i],[]).append(i)
        chunks=[[f for f in c if f not in SEG] for c in chunks]; chunks=[c for c in chunks if c]
        for (sid,s0,s1),fs in segs.items():
            c=(s0+s1)//2; ch=list(range(max(s0,c-NB//2),min(s1,c-NB//2+NB))); chunks.append(ch); reuse[tuple(ch)]=(c,fs)
    for ch in chunks:
        if a_.tiny and rname!='static' and max(int((hf(i)>0).sum()) for i in ch)<a_.tiny:
            for i in ch:
                F=get_out(i); m=cv2.dilate(hf(i),np.ones((3,3),np.uint8))
                if m.any(): F=cv2.inpaint(F,m,3,cv2.INPAINT_TELEA)
                out_full[i]=cv2.imencode('.jpg',F,[cv2.IMWRITE_JPEG_QUALITY,96])[1]
            stats['tiny']=stats.get('tiny',0)+1; continue
        if not any(hf(i).any() for i in ch):
            for i in ch: out_full.setdefault(i,cv2.imencode('.jpg',FRP(i),[cv2.IMWRITE_JPEG_QUALITY,96])[1])
            stats['skipped']=stats.get('skipped',0)+1; continue
        s_ids=np.nonzero(shot==shot[ch[0]])[0]; lo,hi=s_ids[0],s_ids[-1]
        nb=ch+[ch[-1]]*(NB-len(ch)); c=(ch[0]+ch[-1])//2
        refs=[int(np.clip(c+o,lo,hi)) for o in OFF]
        ids=nb+refs
        # rect: union of masks over the chunk (+margin)
        U=np.zeros((H,W),np.uint8)
        for i in ch: U|=hf(i)
        ys,xs=np.nonzero(U); mg=12
        rect=(max(0,xs.min()-mg),max(0,ys.min()-mg),min(W,xs.max()+mg+1),min(H,ys.max()+mg+1))
        stats.setdefault('left_'+rname,[]).append(max(int((hf(i)>0).sum()) for i in ch))
        rot,k,pieces=layout(rect)
        stats.setdefault('scale_'+rname,[]).append(round(MW_/(pieces[0][1][2]-pieces[0][1][0]),2))
        fr=np.stack([canvas(FRP(i),rot,k,pieces) for i in ids]).astype(np.float32)/255
        mk=np.stack([cv2.dilate(canvas(hf(i),rot,k,pieces),np.ones((3,3),np.uint8)) for i in ids]).astype(np.float32)/255
        fr=fr[...,::-1].transpose(0,3,1,2).copy(); mk=(mk>0.5).astype(np.float32)[:,None]
        t1=time.time(); o=sess.run(None,{'frames':fr,'masks':mk})[0]; stats['infer_s']+=time.time()-t1; stats['calls']+=1
        o=(np.clip(o.transpose(0,2,3,1)[...,::-1],0,1)*255).astype(np.uint8)
        plist=list(enumerate(ch))
        if tuple(ch) in reuse:
            c_,fs=reuse[tuple(ch)]; qc=ch.index(c_) if c_ in ch else len(ch)//2; plist=[(qc,i) for i in fs]
        for q,i in plist:
            F=get_out(i); vF=view(F,rot).copy(); vM=view(mf(i),rot)
            for j,(core,ctx) in enumerate(pieces):
                bx,by,bx2,by2=ctx; tile=cv2.resize(o[q][j*MH_//k:(j+1)*MH_//k],(bx2-bx,by2-by),interpolation=cv2.INTER_CUBIC)
                cx0,cy0,cx1,cy1=core
                # restrict paste to this piece's core columns (rows: full ctx height)
                sl=(slice(by,by2),slice(max(bx,cx0 if k>1 else bx),min(bx2,cx1 if k>1 else bx2)))
                hv=(view(hf(i),rot)[sl]>0).astype(np.uint8)
                m=hv
                soft=cv2.GaussianBlur(cv2.dilate(m,np.ones((9,9),np.uint8)).astype(np.float32),(9,9),0); soft=np.maximum(soft,m)[...,None]   # weight 1 on the whole hole, feather outside it
                tl=tile[:,sl[1].start-bx:sl[1].stop-bx]
                blend=(tl*soft+vF[sl]*(1-soft)).astype(np.uint8)
                # clean-up: tiny bright specks the model copies from overlay remnants -> small Telea fill
                gg=cv2.cvtColor(blend,cv2.COLOR_BGR2GRAY); th_=cv2.morphologyEx(gg,cv2.MORPH_TOPHAT,np.ones((9,9),np.uint8))
                sp=((th_>35)&(hv>0)).astype(np.uint8)
                if sp.any():
                    n_,lab_,ss_,_=cv2.connectedComponentsWithStats(sp); bad=np.zeros_like(sp)
                    for k_ in range(1,n_):
                        if ss_[k_,4]<=80: bad[lab_==k_]=255
                    if bad.any(): blend=cv2.inpaint(blend,cv2.dilate(bad,np.ones((5,5),np.uint8)),3,cv2.INPAINT_TELEA)
                vF[sl]=blend
            out_full[i]=cv2.imencode('.jpg',unview(vF,rot),[cv2.IMWRITE_JPEG_QUALITY,96])[1]
    print(rname,'chunks',flush=True) if False else print(rname,'chunks',len(chunks),'%.1fs'%(time.time()-t))
tm['sttn_total']=time.time()-t; tm.update(stats)
np.save(a_.out+'_frames.npy',np.array(sorted(out_full)))
for i in want: out_full.setdefault(i,cv2.imencode('.jpg',FRP(i),[cv2.IMWRITE_JPEG_QUALITY,96])[1])
if a_.frames=='all':
    vw=cv2.VideoWriter(a_.out+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
    for i in range(N): vw.write(get_out(i))
    vw.release()
else:
    for i in out_full: cv2.imwrite(f'{a_.out}_{i:04d}.png',get_out(i))
# also dump the masks used for the frames written (for the comparison sheet)
for i in sorted(out_full)[::1]:
    if a_.frames!='all' or i in (90,240,420,660):
        cv2.imwrite(f'{a_.out}_mask_{i:04d}.png',full_subhole(i)|statH|cv2.dilate(movm(i),_mvK))
tm['total']=time.time()-T0
print(json.dumps({k:round(v,1) if isinstance(v,float) else v for k,v in tm.items()}))
json.dump(tm,open(a_.out+'_timing.json','w'))
