# In-hole quality metrics: PSNR / SSIM restricted to hole pixels, Laplacian-variance sharpness ratio (output / ground truth, 1 = same detail).
import numpy as np, cv2
from skimage.metrics import structural_similarity
def psnr(out, gt, m):
    m=m>0
    if not m.any(): return float('nan')
    d=(out.astype(np.float32)-gt.astype(np.float32))[m]; mse=float((d*d).mean())
    return 99.0 if mse<1e-10 else 10*np.log10(255.0**2/mse)
def ssim(out, gt, m):
    m=m>0
    if not m.any(): return float('nan')
    ys,xs=np.nonzero(m); p=8; y0,y1,x0,x1=max(0,ys.min()-p),ys.max()+p+1,max(0,xs.min()-p),xs.max()+p+1
    _,S=structural_similarity(gt[y0:y1,x0:x1],out[y0:y1,x0:x1],channel_axis=2,full=True,data_range=255)
    return float(S.mean(2)[m[y0:y1,x0:x1]].mean())
def lapvar(img, m):
    m=m>0
    if not m.any(): return float('nan')
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32); L=cv2.Laplacian(g,cv2.CV_32F,ksize=3)
    return float(L[cv2.erode(m.astype(np.uint8),np.ones((3,3),np.uint8))>0].var())
def all3(out, gt, m):
    lg=lapvar(gt,m)
    return dict(psnr=psnr(out,gt,m),ssim=ssim(out,gt,m),sharp=lapvar(out,m)/max(lg,1e-6))
def mean(rows):
    ks=rows[0].keys() if rows else []
    return {k:float(np.nanmean([r[k] for r in rows])) for k in ks}
