import cv2, numpy as np, sys
pref=sys.argv[1]; tag=sys.argv[2]; label=sys.argv[3] if len(sys.argv)>3 else 'STTN'
mw=np.load('moving_wm.npz')['pos']
def grab(path,i,size=(1080,1920)):
    c=cv2.VideoCapture(path); c.set(1,i); _,f=c.read(); return cv2.resize(f,size,interpolation=cv2.INTER_CUBIC) if f.shape[1]!=size[0] else f
def hdr(names,w):
    h=np.full((44,w*len(names),3),20,np.uint8)
    for k,n in enumerate(names): cv2.putText(h,n,(k*w+10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(151,220,61),2)
    return h
names=['source','v0.2',label,'mini-program']
for t,i in [(3,90),(8,240),(14,420),(22,660)]:
    src=grab('/workspace/subremover/sample/a.mp4',i); v02=grab('/workspace/subremover/test/o3_a.mp4',i)
    st=cv2.imread(f'{pref}_{i:04d}.png'); mp=grab('/workspace/subremover/sample/b.mp4',i)
    mask=cv2.imread(f'{pref}_mask_{i:04d}.png',0)
    imgs=[src,v02,st,mp]
    # overview with mask outline on the source
    so=src.copy(); cnt,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(so,cnt,-1,(0,0,255),3)
    ov=np.hstack([cv2.resize(x,(405,720),interpolation=cv2.INTER_AREA) for x in [so]+imgs[1:]])
    x,y=int(mw[i,0]),int(mw[i,1]); mx0=int(np.clip(x-60,0,1080-300)); my0=int(np.clip(y-50,0,1920-180))
    zs=[]
    for x0,y0,x1,y1,sc in [(140,1400,940,1600,0.5),(990,40,1070,600,1.0),(mx0,my0,mx0+300,my0+180,1.35)]:
        tiles=[x[y0:y1,x0:x1] for x in imgs]
        if y1-y0>x1-x0: tiles=[cv2.rotate(tt,cv2.ROTATE_90_COUNTERCLOCKWISE) for tt in tiles]
        row=np.hstack(tiles); s=1620/row.shape[1]; zs.append(cv2.resize(row,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC))
    sheet=np.vstack([hdr(names,405),ov,hdr(['zoom: subtitle (full-res)'],1620),zs[0],hdr(['zoom: fixed top-right watermark (rotated, full-res)'],1620),zs[1],hdr(['zoom: moving watermark (full-res)'],1620),zs[2]])
    cv2.imwrite(f'/workspace/subremover/compare/{tag}_{t:02d}s.png',sheet)
    print(tag,t,sheet.shape)
