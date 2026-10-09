import sys,time
sys.argv=['pipe.py','--frames','0:1']
src=open('pipe.py').read(); src=src[:src.index('sess_o=')]
exec(src)
import prop
_k5=np.ones((7,7),np.uint8); _k13=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
def U(i): return cv2.dilate(full_sub(i),_k5)|statH|cv2.dilate(movm(i),_mvK)
_k15=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
def BAD(i):
    hsv=cv2.cvtColor(bands[i],cv2.COLOR_BGR2HSV); g=hsv[...,2]; th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,hatK)
    r=((th>35)&(g>170)&(hsv[...,1]<80)).astype(np.uint8)*255; r[:,:W//2-int(W*0.42)]=0; r[:,W//2+int(W*0.42):]=0
    m=np.zeros((H,W),np.uint8); m[y0:y1]=cv2.dilate(r,_k15); return m|cv2.dilate(U(i),_k5)
rows=[]
for t in (90,356,600,840):
    s_ids=np.nonzero(shot==shot[t])[0]; lo,hi=max(s_ids[0],t-20),min(s_ids[-1],t+20)
    ids=list(range(lo,hi+1)); t0=time.time()
    out,un=prop.propagate([FR(i) for i in ids],[U(i) for i in ids],srcbad=[BAD(i) for i in ids])
    k=ids.index(t); h=U(t); print(t,'range',lo,hi,'%.1fs'%(time.time()-t0),'hole px',int((h>0).sum()),'unseen px',int((un[k]>0).sum()))
    cv2.imwrite(f'o/prop_{t:04d}.png',out[k]); cv2.imwrite(f'o/prop_un_{t:04d}.png',un[k])
