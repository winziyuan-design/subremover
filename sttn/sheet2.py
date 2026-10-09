import cv2,numpy as np,sys
A,B,name=sys.argv[1],sys.argv[2],sys.argv[3]; LA,LB=sys.argv[4],sys.argv[5]
P=np.load('wm_pos_refined.npy')
def grab(p,i):
    c=cv2.VideoCapture(p); c.set(1,i); _,f=c.read(); return cv2.resize(f,(1080,1920),interpolation=cv2.INTER_CUBIC)
rows=[]
for i in [int(x) for x in sys.argv[6].split(',')]:
    a=grab('/workspace/subremover/sample/a.mp4',i); b=grab('/workspace/subremover/sample/b.mp4',i)
    o1=cv2.imread(f'{A}_{i:04d}.png'); o2=cv2.imread(f'{B}_{i:04d}.png')
    x,y=P[i]; x0=int(np.clip(x-20,0,1080-250)); y0=int(np.clip(y-15,0,1920-160))
    ims=(a,o1,o2,b)
    r1=np.hstack([im[1420:1580,250:830] for im in ims])
    r2=np.hstack([cv2.resize(im[y0:y0+160,x0:x0+250],(580,371)) for im in ims])
    r=np.vstack([r1,r2]); cv2.putText(r,f'f{i} ({i/30:.1f}s)',(8,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(151,220,61),2); rows.append(r)
h=np.full((40,rows[0].shape[1],3),(16,12,10),np.uint8)
for k,l in enumerate(['source',LA,LB,'mini-program']): cv2.putText(h,l,(k*580+8,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(151,220,61),2)
img=np.vstack([h]+rows); cv2.imwrite(f'/workspace/subremover/compare/{name}.png',img); print(img.shape)
