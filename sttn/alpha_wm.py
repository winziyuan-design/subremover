# Semi-transparent moving watermark: estimate per-pixel opacity from many aligned frames, then reverse the blend
import cv2, numpy as np, time
t0=time.time()
d=dict(np.load('moving_wm.npz')); pos=d['pos']; sc=d['score']; M=d['mask']; h,w=M.shape
t=cv2.imread('wm_t3.png',0); med=cv2.resize(t[:, :t.shape[1]//2],None,fx=0.5,fy=0.5).astype(np.uint8)
K=np.ones((11,11),np.uint8); tp=cv2.morphologyEx(med,cv2.MORPH_TOPHAT,K)
cap=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); N=int(cap.get(7))
P=np.zeros((N,2),int); crops=[]; ests=[]
core=cv2.erode(M,np.ones((5,5),np.uint8))>0
for i in range(N):
    _,f=cap.read(); x,y=pos[i].round().astype(int)
    g=cv2.morphologyEx(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.MORPH_TOPHAT,K)
    R=5; x0,y0=max(0,x-R),max(0,y-R); sub=g[y0:y+h+R,x0:x+w+R]
    if sub.shape[0]>=h and sub.shape[1]>=w:
        r=cv2.matchTemplate(sub,tp,cv2.TM_CCOEFF_NORMED); _,m,_,l=cv2.minMaxLoc(r); x,y=x0+l[0],y0+l[1]
    P[i]=(x,y)
    if i%2==0 and sc[i]>0.4 and 0<=y and y+h<=f.shape[0] and 0<=x and x+w<=f.shape[1]:
        c=f[y:y+h,x:x+w].astype(np.float32).max(2)   # brightest channel (white overlay raises all channels)
        bg=cv2.inpaint(f[y:y+h,x:x+w],(M>0).astype(np.uint8)*255,5,cv2.INPAINT_TELEA).astype(np.float32).max(2)
        a=(c-bg)/np.maximum(255-bg,20); ests.append(a)
A=np.clip(np.median(ests,0),0,0.95); A[M==0]=0
A=cv2.GaussianBlur(A,(3,3),0)
print('frames used',len(ests),'alpha p50/p90/max in mask %.2f %.2f %.2f'%(np.median(A[core]),np.percentile(A[core],90),A.max()),'%.1fs'%(time.time()-t0))
np.save('wm_alpha.npy',A); np.save('wm_pos_refined.npy',P)
cv2.imwrite('wm_alpha.png',cv2.resize((A*255/A.max()).astype(np.uint8),None,fx=3,fy=3))
