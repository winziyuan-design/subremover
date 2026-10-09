import sys
sys.argv=['pipe.py','--frames','0:1']
src=open('pipe.py').read(); src=src[:src.index('sess_o=')]
exec(src)
ring=cv2.dilate(statH,np.ones((31,31),np.uint8))&~cv2.dilate(statH,np.ones((7,7),np.uint8))
ys,xs=np.nonzero(statH); print('static bbox',xs.min(),ys.min(),xs.max(),ys.max())
for s in range(shot[-1]+1):
    ids=np.nonzero(shot==s)[0]; g=np.stack([cv2.cvtColor(FR(i),cv2.COLOR_BGR2GRAY)[ring>0].astype(np.float32) for i in ids[::2]])
    d=np.abs(g-g[len(g)//2]).mean(1)
    print(s,ids[0],ids[-1],'ring mean abs diff vs mid: max %.1f median %.1f'%(d.max(),np.median(d)))
