import cv2, numpy as np, glob, json
det=cv2.dnn_TextDetectionModel_DB('/workspace/subremover/third_party/det.onnx'); det.setBinaryThreshold(0.3); det.setPolygonThreshold(0.5); det.setMaxCandidates(200); det.setUnclipRatio(1.6)
MEAN=(122.67891434,116.66876762,104.00698793)
def ntext(img):
    h,w=img.shape[:2]; s=1.5; det.setInputParams(1/255,(max(32,int(w*s)//32*32),max(32,int(h*s)//32*32)),MEAN)
    return len([p for p in det.detect(img)[0] if cv2.contourArea(np.array(p,np.float32))>150])
P=np.load('wm_pos_refined.npy')
ca=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); cb=cv2.VideoCapture('/workspace/subremover/sample/b.mp4')
fr=sorted(int(p[-8:-4]) for p in glob.glob('o/s16_[0-9][0-9][0-9][0-9].png'))
res={'sub':[0,0,0],'static':[0,0,0],'moving':[0,0,0]}; sharp={'sub':[],'moving':[]}; bad=[]
for i in fr:
    ca.set(1,i); cb.set(1,i); _,a=ca.read(); _,b=cb.read(); b=cv2.resize(b,(1080,1920),interpolation=cv2.INTER_CUBIC); o=cv2.imread(f'o/s16_{i:04d}.png')
    m=cv2.imread(f'o/s16_mask_{i:04d}.png',0) if False else None
    x,y=P[i]; regs={'sub':(slice(1426,1567),slice(330,750)),'static':(slice(40,600),slice(985,1075)),'moving':(slice(max(0,y-10),y+135),slice(max(0,x-10),x+215))}
    for k,sl in regs.items():
        na,nb,no=ntext(a[sl]),ntext(b[sl]),ntext(o[sl]); res[k][0]+=na>0; res[k][1]+=no>0; res[k][2]+=nb>0
        if no>nb and na>0: bad.append((i,k))
    for k in ('sub','moving'):
        sl=regs[k]; lv=lambda im: cv2.Laplacian(cv2.cvtColor(im[sl],cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
        sharp[k].append((lv(o),lv(b),lv(a)))
print('frames evaluated',len(fr))
for k,(na,no,nb) in res.items(): print(f'{k}: frames with text detected  source={na}  ours={no}  mini-program={nb}')
for k,v in sharp.items():
    v=np.array(v); print(f'{k}: median Laplacian var (detail) ours={np.median(v[:,0]):.0f} mini={np.median(v[:,1]):.0f} source={np.median(v[:,2]):.0f}')
print('frames where ours still shows text the mini-program removed:',len(bad), bad[:40])
