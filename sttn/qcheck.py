import onnxruntime as ort, numpy as np, cv2
c=cv2.VideoCapture('/workspace/subremover/sample/a.mp4'); fr=[];mk=[]
for i in range(85,101):
    c.set(1,i); _,f=c.read(); crop=f[1380:1620,140:940]; crop=cv2.resize(crop,(432,240),interpolation=cv2.INTER_AREA)
    fr.append(crop[...,::-1].transpose(2,0,1)/255.); m=np.zeros((240,432),np.float32); m[60:180,60:372]=1; mk.append(m[None])
fr=np.array(fr,np.float32); mk=np.array(mk,np.float32)
so=ort.SessionOptions(); so.intra_op_num_threads=2
outs={}
for n in ['sttn_t16.onnx','sttn_t16_fp16.onnx','sttn_t16_int8.onnx']:
    outs[n]=ort.InferenceSession(n,so).run(None,{'frames':fr,'masks':mk})[0]
ref=outs['sttn_t16.onnx']; hole=np.broadcast_to(mk>0,ref.shape)
for n,o in outs.items():
    mse=((o-ref)[hole]**2).mean(); print(n,'PSNR in hole vs fp32: %.1f dB'%(10*np.log10(1/max(mse,1e-12))))
row=lambda o:(np.clip(o[5].transpose(1,2,0)[...,::-1],0,1)*255).astype(np.uint8)
cv2.imwrite('qcheck.png',np.vstack([row(fr*(1-mk))]+[row(o) for o in outs.values()]))
