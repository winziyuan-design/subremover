# In-hole PSNR / SSIM / sharpness on make_eval.py clips for a torch checkpoint (--ckpt, netG) or an exported ONNX (--onnx); --telea for a non-learned baseline.
import os, sys, glob, json, argparse, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.dirname(HERE))
import metrics
def evaluate(run, d, limit=0):
    fs=sorted(glob.glob(os.path.join(d,'*.npz')))[:limit or None]; rows=[]
    for f in fs:
        z=np.load(f); gt,inp,mk=z['gt'],z['inp'],z['mask']
        if run is None: out=np.stack([cv2.inpaint(x,cv2.dilate(m,np.ones((3,3),np.uint8)),3,cv2.INPAINT_TELEA) for x,m in zip(inp,mk)])
        else:
            fr=(inp[...,::-1].astype(np.float32)/255).transpose(0,3,1,2).copy(); m=(mk>0).astype(np.float32)[:,None]
            o=np.clip(run(fr,m).transpose(0,2,3,1)[...,::-1],0,1)*255
            out=np.clip(o*m.transpose(0,2,3,1)+inp*(1-m.transpose(0,2,3,1))+0.5,0,255).astype(np.uint8)
        rows+=[metrics.all3(o_,g_,m_) for o_,g_,m_ in zip(out,gt,mk) if m_.any()]
    r=metrics.mean(rows); r['clips']=len(fs); r['frames']=len(rows); return r
if __name__=='__main__':
    import sttn_net
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--ckpt'); ap.add_argument('--onnx'); ap.add_argument('--telea',action='store_true')
    ap.add_argument('--sttn-repo',default=sttn_net.STTN_REPO); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--json'); a=ap.parse_args()
    run=None if a.telea else sttn_net.runner(None if a.onnx else sttn_net.generator(a.sttn_repo,a.ckpt or sttn_net.STTN_CKPT),a.onnx,a.threads)
    r=evaluate(run,a.data,a.limit); r['model']=('telea' if a.telea else a.onnx or a.ckpt or sttn_net.STTN_CKPT); print(json.dumps(r))
    if a.json: json.dump(r,open(a.json,'w'),indent=1)
