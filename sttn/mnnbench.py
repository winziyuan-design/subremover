import MNN, numpy as np, time, onnxruntime as ort
np.random.seed(0); fr=np.random.rand(16,3,240,432).astype(np.float32); mk=np.zeros((16,1,240,432),np.float32); mk[:,:,150:200,100:300]=1
so=ort.SessionOptions(); so.intra_op_num_threads=4; so.enable_cpu_mem_arena=False
s=ort.InferenceSession('sttn_t16.onnx',so); ref=s.run(None,{'frames':fr,'masks':mk})[0]
t=time.time(); s.run(None,{'frames':fr,'masks':mk}); s.run(None,{'frames':fr,'masks':mk}); print('ORT 4thr %.2fs'%((time.time()-t)/2)); del s
import MNN.expr as F
for fn in ('sttn_t16.mnn','sttn_t16_fp16.mnn'):
    rt=MNN.nn.create_runtime_manager(({'backend':'CPU','precision':'normal','numThread':4},))
    net=MNN.nn.load_module_from_file(fn,['frames','masks'],['out'],runtime_manager=rt)
    a=F.const(fr,[16,3,240,432],F.NCHW); b=F.const(mk,[16,1,240,432],F.NCHW)
    o=net.forward([a,b])[0]; o=F.convert(o,F.NCHW); o=np.array(o.read()).reshape(ref.shape)
    t=time.time(); [np.array(F.convert(net.forward([a,b])[0],F.NCHW).read()) for _ in range(2)]
    print(fn,'MNN cpu 4thr %.2fs'%((time.time()-t)/2),'maxdiff %.4f'%np.abs(o-ref).max())
