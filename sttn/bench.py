import onnxruntime as ort, numpy as np, time, sys, resource
p=sys.argv[1]; so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession(p,so,providers=['CPUExecutionProvider'])
T=s.get_inputs()[0].shape[0]; fr=np.random.rand(T,3,240,432).astype(np.float32); mk=np.zeros((T,1,240,432),np.float32); mk[:,:,150:200,100:300]=1
o=s.run(None,{'frames':fr,'masks':mk})[0]
t=time.time(); n=2
for _ in range(n): o=s.run(None,{'frames':fr,'masks':mk})[0]
dt=(time.time()-t)/n
print('%s T=%d %.2fs/call %.3fs/frame maxrss %dMB'%(p,T,dt,dt/T,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024))
