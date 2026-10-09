import cv2, numpy as np
ca=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); cb=cv2.VideoCapture('/workspace/subremover/sample/b.mp4')
det = cv2.dnn_TextDetectionModel_DB('/workspace/subremover/third_party/det.onnx')
det.setBinaryThreshold(0.3); det.setPolygonThreshold(0.5); det.setMaxCandidates(200); det.setUnclipRatio(2.0)
det.setInputParams(1.0/255, (544, 960), (122.67891434, 116.66876762, 104.00698793))
tiles=[]
for t in [0.5,4,8,12,16,20,24,27]:
    ca.set(0,t*1000); cb.set(0,t*1000); _,a=ca.read(); _,b=cb.read(); b=cv2.resize(b,(1080,1920))
    d=cv2.cvtColor(cv2.absdiff(a,b),cv2.COLOR_BGR2GRAY); d=cv2.GaussianBlur(d,(5,5),0)
    polys,_=det.detect(a); v=a.copy()
    for p in polys: cv2.polylines(v,[np.array(p,np.int32)],True,(0,0,255),4)
    v[d>40]=(0,255,0)
    tiles.append(cv2.resize(v,(360,640)))
cv2.imwrite('look.jpg',np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
