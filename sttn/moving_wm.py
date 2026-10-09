import cv2, numpy as np, time
t0=time.time()
cap=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); N=int(cap.get(7))
det = cv2.dnn_TextDetectionModel_DB('/workspace/subremover/third_party/det.onnx')
det.setBinaryThreshold(0.3); det.setPolygonThreshold(0.5); det.setMaxCandidates(200); det.setUnclipRatio(1.6)
det.setInputParams(1.0/255, (544, 960), (122.67891434, 116.66876762, 104.00698793))
SZ=(64,32)
def desc(g, r):
    x,y,w,h=r; c=g[max(0,y):y+h, max(0,x):x+w]
    if c.size==0: return None
    c=cv2.resize(c,SZ).astype(np.float32); c-=c.mean(); n=np.linalg.norm(c); return c/n if n>0 else None
boxes=[]  # (frameidx, rect, desc)
frames_s=list(range(0,N,6))
for i in frames_s:
    cap.set(1,i); _,f=cap.read(); g=cv2.morphologyEx(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.MORPH_TOPHAT,np.ones((11,11),np.uint8))
    for p in det.detect(f)[0]:
        r=cv2.boundingRect(np.array(p,np.int32)); x,y,w,h=r
        if w<30 or h<15 or w>500 or (y+h>1380 and y<1620) or x>980: continue
        d=desc(g,r)
        if d is not None and g[y:y+h,x:x+w].mean()>12: boxes.append((i,r,d))
print(len(boxes),'boxes',sorted([(b[0],b[1]) for b in boxes])[:0], '%.1fs'%(time.time()-t0))
D=np.stack([b[2].ravel() for b in boxes]); S=D@D.T
W=np.array([b[1][2] for b in boxes]); H=np.array([b[1][3] for b in boxes]); F=np.array([b[0] for b in boxes])
best=None
for k in range(len(boxes)):
    ok=(S[k]>0.55)&(np.abs(W/W[k]-1)<0.2)&(np.abs(H/H[k]-1)<0.2)
    fr=set(F[ok]); X=np.array([boxes[j][1][0] for j in np.nonzero(ok)[0]]); Y=np.array([boxes[j][1][1] for j in np.nonzero(ok)[0]])
    spread=np.ptp(X)+np.ptp(Y)  # must move (static handled elsewhere)
    sc=len(fr)
    if spread>200 and (best is None or sc>best[0]): best=(sc,k,ok)
sc,k,ok=best; print('template support', sc,'/',len(frames_s),'rect',boxes[k][1])
# median aligned crop (use exemplar size)
w0,h0=int(boxes[k][1][2]*1.3)//2*2,int(boxes[k][1][3]*1.9)//2*2
crops=[]
for j in np.nonzero(ok)[0]:
    i,(x,y,w,h),_=boxes[j]; cap.set(1,i); _,f=cap.read(); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    cx,cy=x+w//2,y+h//2; c=g[max(0,cy-h0//2):cy-h0//2+h0, max(0,cx-w0//2):cx-w0//2+w0]
    if c.shape==(h0,w0): crops.append(c)
med=np.median(crops,0).astype(np.uint8)
cv2.imwrite('wm_template.png',cv2.resize(med,None,fx=3,fy=3))
np.save('wm_template.npy',med)
print('done %.1fs'%(time.time()-t0))
# ---- track through every frame by template matching on white top-hat (half res) ----
def th(f):
    t=cv2.morphologyEx(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.MORPH_TOPHAT,np.ones((11,11),np.uint8)); t[1400:1600]=0; t[40:600,980:]=0; return t
def track(tpl):
    tp=cv2.resize(cv2.morphologyEx(tpl,cv2.MORPH_TOPHAT,np.ones((11,11),np.uint8)),None,fx=0.5,fy=0.5)
    cap.set(1,0); pos=[]; sc=[]; prev=None
    for i in range(N):
        _,f=cap.read(); g=cv2.resize(th(f),None,fx=0.5,fy=0.5)
        if prev is not None:
            R=60; x0=max(0,prev[0]-R); y0=max(0,prev[1]-R); sub=g[y0:prev[1]+tp.shape[0]+R, x0:prev[0]+tp.shape[1]+R]
            r=cv2.matchTemplate(sub,tp,cv2.TM_CCOEFF_NORMED); _,m,_,l=cv2.minMaxLoc(r); l=(l[0]+x0,l[1]+y0)
        if prev is None or m<0.35:
            r=cv2.matchTemplate(g,tp,cv2.TM_CCOEFF_NORMED); _,m,_,l=cv2.minMaxLoc(r)
        pos.append((l[0]*2,l[1]*2)); sc.append(m); prev=l if m>0.35 else prev
    return np.array(pos,float), np.array(sc)
pos,sc=track(med)
print('track1 score median %.2f, <0.35: %d frames  %.1fs'%(np.median(sc),(sc<0.35).sum(),time.time()-t0))
# refine template from confidently tracked frames
cap.set(1,0); crops=[]
for i in range(N):
    _,f=cap.read()
    if sc[i]>0.5 and i%3==0:
        x,y=map(int,pos[i]); c=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)[y:y+h0,x:x+w0]
        if c.shape==(h0,w0): crops.append(c)
med2=np.median(crops,0).astype(np.uint8); print('refine from',len(crops))
pos,sc=track(med2)
# robust smoothing: watermark motion is smooth -> fit local linear model, drop outliers
good=sc>0.4; idx=np.arange(N); sm=np.zeros_like(pos)
for i in range(N):
    w=(np.abs(idx-i)<=90)&good
    A=np.stack([idx[w]-i,np.ones(w.sum())],1); P=pos[w]; keep=np.ones(len(P),bool)
    for it in range(4):
        c=np.linalg.lstsq(A[keep],P[keep],rcond=None)[0]; r=np.abs(A@c-P).max(1); keep=r<max(8,np.median(r[keep])*2.5)
    sm[i]=c[1]
print('track2 score median %.2f, low: %d, max |raw-smooth| on good %.1f  %.1fs'%(np.median(sc),(sc<0.3).sum(),np.abs(pos-sm)[good].max(),time.time()-t0))
# stroke mask from refined template: bright strokes over the median background
tm=cv2.morphologyEx(med2,cv2.MORPH_TOPHAT,np.ones((11,11),np.uint8))
smask=((tm>np.percentile(tm,75))&(tm>8)).astype(np.uint8)*255
smask=cv2.dilate(smask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
cv2.imwrite('wm_template2.png',cv2.resize(np.hstack([med2,smask]),None,fx=2,fy=2))
print([(i,tuple(sm[i].astype(int)),round(float(sc[i]),2)) for i in range(0,N,20)])
np.savez('moving_wm.npz',pos=sm,score=sc,mask=smask)
