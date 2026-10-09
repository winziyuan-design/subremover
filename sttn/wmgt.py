# analysis only: where does the moving watermark sit (from |a-b|), and what does DB detect there
import cv2, numpy as np
ca=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); cb=cv2.VideoCapture('/workspace/subremover/sample/b.mp4')
det = cv2.dnn_TextDetectionModel_DB('/workspace/subremover/third_party/det.onnx')
det.setBinaryThreshold(0.3); det.setPolygonThreshold(0.5); det.setMaxCandidates(200); det.setUnclipRatio(1.6)
det.setInputParams(1.0/255, (544, 960), (122.67891434, 116.66876762, 104.00698793))
for i in range(0,856,40):
    ca.set(1,i); cb.set(1,i); _,a=ca.read(); _,b=cb.read(); b=cv2.resize(b,(1080,1920))
    d=cv2.GaussianBlur(cv2.cvtColor(cv2.absdiff(a,b),cv2.COLOR_BGR2GRAY),(5,5),0)>40
    d[1350:1650]=0; d[:,980:]=0
    ys,xs=np.nonzero(d)
    gt=(xs.min(),ys.min(),xs.max(),ys.max()) if len(xs)>50 else None
    bx=[cv2.boundingRect(np.array(p,np.int32)) for p in det.detect(a)[0]]
    hit=[r for r in bx if gt and r[0]<gt[2] and r[0]+r[2]>gt[0] and r[1]<gt[3] and r[1]+r[3]>gt[1]]
    print(i, gt, hit)
