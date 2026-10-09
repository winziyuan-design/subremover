import sys
sys.argv=['pipe.py','--frames','0:1']
src=open('pipe.py').read(); src=src[:src.index('sess_o=')]
exec(src)
cb=cv2.VideoCapture('/workspace/subremover/sample/b.mp4'); bad=[]; tot=[]
for i in range(N):
    _,b=cb.read(); b=cv2.resize(b,(W,H),interpolation=cv2.INTER_CUBIC)
    a=FR(i); d=cv2.GaussianBlur(cv2.cvtColor(cv2.absdiff(a,b),cv2.COLOR_BGR2GRAY),(5,5),0)>45
    hole=full_subhole(i)|statH|cv2.dilate(movm(i),_mvK)
    miss=d&(cv2.dilate(hole,np.ones((9,9),np.uint8))==0)
    sb=miss[y0:y1].sum(); st_=miss[:700,900:].sum(); tot.append((d[y0:y1].sum(),sb))
    if sb>400: bad.append((i,int(sb)))
print('frames with >400 changed-but-unmasked px in subtitle band:',len(bad)); print(bad[:60])
