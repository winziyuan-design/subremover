import cv2, numpy as np
cap=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); N=int(cap.get(7))
ed=[]; br=[]
for i in np.linspace(0,N-1,60).astype(int):
    cap.set(1,i); _,f=cap.read(); g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    ed.append(cv2.Canny(g,60,160)>0)
    br.append(cv2.morphologyEx(g,cv2.MORPH_TOPHAT,np.ones((15,15),np.uint8)))
ef=np.mean(ed,0); th=np.median(br,0)
cv2.imwrite('ef.png',(ef*255).astype(np.uint8)); cv2.imwrite('th.png',th)
m=((ef>0.5)|(th>60)).astype(np.uint8)*255
cv2.imwrite('static_raw.png',m)
n,lab,st,_=cv2.connectedComponentsWithStats(cv2.dilate(m,np.ones((15,15),np.uint8)))
for k in range(1,n): print(st[k])
