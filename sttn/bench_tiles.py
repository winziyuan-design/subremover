# Ground-truth benchmark: synthetic subtitles + moving watermark over clean clips; legacy 432x240 canvas (A) vs native tiles (B), each with/without flow propagation, same STTN weights.
import os, sys, json, time, argparse, cv2, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'train'))
import sttn_net, tiles, prop, synth, metrics
ap=argparse.ArgumentParser()
ap.add_argument('videos',nargs='+'); ap.add_argument('--ckpt',default=sttn_net.STTN_CKPT); ap.add_argument('--repo',default=sttn_net.STTN_REPO); ap.add_argument('--onnx')
ap.add_argument('--chunks',type=int,default=2); ap.add_argument('--nb',type=int,default=10); ap.add_argument('--T',type=int,default=16)
ap.add_argument('--variants',default='telea,A,B,propA,propB'); ap.add_argument('--R',type=int,default=12); ap.add_argument('--crop-w',type=int,default=1080); ap.add_argument('--ov',type=int,default=96); ap.add_argument('--ctx',type=int,default=64)
ap.add_argument('--seed',type=int,default=0); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--fonts',nargs='*')
ap.add_argument('--out',default='bench_out'); ap.add_argument('--sheet',default=None); a=ap.parse_args()
os.makedirs(a.out,exist_ok=True)
run=sttn_net.runner(None if a.onnx else sttn_net.generator(a.repo,a.ckpt or None),a.onnx,a.threads)
fonts=synth.find_fonts(a.fonts); V=a.variants.split(','); NR=a.T-a.nb
CROP=[0,None,0,None]
def decode(p):
    cap=cv2.VideoCapture(p); fps=cap.get(5) or 30; js=[]; rows=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        if len(js)%10==0: rows.append(f.mean((1,2)))
        js.append(cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,97])[1])
    keep=np.nonzero(np.mean(rows,0)>6)[0]; h,w=cv2.imdecode(js[0],cv2.IMREAD_COLOR).shape[:2]
    cw=min(w,a.crop_w or w); CROP[:]=[keep[0],keep[-1]+1,(w-cw)//2,(w-cw)//2+cw] if len(keep) else [0,h,(w-cw)//2,(w-cw)//2+cw]
    return js,fps
def dec(b): y0,y1,x0,x1=CROP; return np.ascontiguousarray(cv2.imdecode(b,cv2.IMREAD_COLOR)[y0:y1,x0:x1])
res={v:[] for v in V}; per={}; timing={v:0.0 for v in V}; infos={v:[] for v in V}; sheet_rows=None
for vi,path in enumerate(a.videos):
    js,fps=decode(path); N=len(js); H,W=dec(js[0]).shape[:2]; st=max(1,int(round(fps/30)))
    rng=np.random.default_rng(a.seed+vi)
    sub=synth.Overlay(W,H,N,rng,fonts,mode='frame',p_wm=0.0)
    wm=synth.Overlay(W,H,N,rng,fonts,mode='frame',p_wm=1.0,p_gap=1.0)
    off=[o*st for o in ([-90,-60,-30,30,60,90] if NR==6 else list(np.linspace(-90,90,NR).astype(int)))]
    _cache={}
    def frame(i):
        if i not in _cache:
            gt=dec(js[i]); x,ms=gt,[]
            for ov in (sub,wm):
                x,m=ov.apply(x,i,dil=max(2,int(H*0.004))); ms.append(m)
            _cache[i]=(gt,x,ms)
        return _cache[i]
    vrow={v:[] for v in V}
    for c in np.linspace(0,N,a.chunks+2)[1:-1].astype(int):
        _cache.clear(); ch=list(range(c-a.nb//2,c-a.nb//2+a.nb)); refs=[int(np.clip(c+o,0,N-1)) for o in off]; ids=ch+refs
        gts=[frame(i)[0] for i in ch]; hole=[frame(i)[2][0]|frame(i)[2][1] for i in ch]
        P=None
        if any(v.startswith('prop') for v in V):
            lo,hi=max(0,ch[0]-a.R),min(N-1,ch[-1]+a.R); pid=list(range(lo,hi+1))
            ph=[cv2.dilate(frame(i)[2][0]|frame(i)[2][1],np.ones((7,7),np.uint8)) for i in pid]
            t=time.time(); po,un=prop.propagate([frame(i)[1] for i in pid],ph); tp=time.time()-t
            P={i:(po[k],cv2.erode(((ph[k]>0)&(un[k]==0)).astype(np.uint8)*255,np.ones((7,7),np.uint8))) for k,i in enumerate(pid)}
        for v in V:
            t=time.time()
            if v=='telea':
                outs=[cv2.inpaint(frame(i)[1],cv2.dilate(frame(i)[2][0]|frame(i)[2][1],np.ones((3,3),np.uint8)),3,cv2.INPAINT_TELEA) for i in ch]
            else:
                usep=v.startswith('prop'); fn=tiles.inpaint_legacy if v.endswith('A') else tiles.inpaint_tiles
                cur=[(P[i][0] if usep and i in P else frame(i)[1]) for i in ids]
                for r in (0,1):
                    hs=[frame(i)[2][r]&~P[i][1] if usep and i in P else frame(i)[2][r] for i in ids]
                    if not any(h.any() for h in hs[:a.nb]): continue
                    info={}
                    kw=dict(ov=a.ov,ctx=a.ctx) if fn is tiles.inpaint_tiles else {}
                    cur=fn(cur,hs,run,info=info,**kw); infos[v].append(info)
                outs=cur[:a.nb]
            dt=time.time()-t+(tp if v.startswith('prop') else 0); timing[v]+=dt
            rows=[metrics.all3(o,g,m) for o,g,m in zip(outs,gts,hole)]; res[v]+=rows; vrow[v]+=rows
            if vi==0 and sheet_rows is None:
                cv2.imwrite(f'{a.out}/v{vi}_c{c}_{v}.png',outs[a.nb//2])
            print(os.path.basename(path),c,v,json.dumps({k:round(x,3) for k,x in metrics.mean(rows).items()}),'%.1fs'%dt,flush=True)
        if vi==0 and sheet_rows is None:
            q=a.nb//2; i=ch[q]; g,x,ms=frame(i); ys,xs=np.nonzero(ms[0]); pad=40
            y0,y1,x0,x1=max(0,ys.min()-pad),min(H,ys.max()+pad),max(0,xs.min()-pad),min(W,xs.max()+pad)
            sheet_rows=[('ground truth',g),('input',x)]+[(v,cv2.imread(f'{a.out}/v{vi}_c{c}_{v}.png')) for v in V]
            sheet_rows=[(n,im[y0:y1,x0:x1]) for n,im in sheet_rows]
    per[os.path.basename(path)]={v:metrics.mean(vrow[v]) for v in V}
summary={v:dict(metrics.mean(res[v]),sec=round(timing[v],1),calls=sum(i.get('calls',0) for i in infos[v])) for v in V}
print(json.dumps(summary,indent=1)); json.dump(dict(summary=summary,per_video=per,args=vars(a)),open(f'{a.out}/bench.json','w'),indent=1)
if a.sheet and sheet_rows:
    rows=[]
    for n,im in sheet_rows:
        im=im.copy(); cv2.putText(im,n,(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),4); cv2.putText(im,n,(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2); rows.append(im)
    cv2.imwrite(a.sheet,np.vstack(rows))
