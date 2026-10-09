import cv2,numpy as np
c=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); LH=58.5; W=1080
hatK=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(LH*0.5)|1,int(LH*0.5)|1))
for i in (252,270):
    c.set(1,i); _,f=c.read(); b=f[1426:1567]
    hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV); g=hsv[...,2]; th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,hatK)
    st=((th>35)&(g>170)&(hsv[...,1]<80))
    cols=st.sum(0); print(i,'active cols near centre', np.nonzero(cols[400:700]>1)[0][[0,-1]]+400 if (cols[400:700]>1).any() else None, 'th max',th[:,450:650].max())
    cv2.imwrite(f'fb{i}.png',np.vstack([b,cv2.cvtColor(th*3,cv2.COLOR_GRAY2BGR),cv2.cvtColor(st.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)]))
