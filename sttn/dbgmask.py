import sys
sys.argv=['pipe.py','--frames','235:245']
src=open('pipe.py').read(); src=src[:src.index('sess_o=')]
exec(src)
for i in list(range(150,340,6))+list(range(400,430,3))+list(range(640,680,4)):
    print(i, 'shot',shot[i],'dets',dets.get(i//K*K),'stroke px',int((subm[i]>0).sum()),'hole px',int((full_subhole(i)>0).sum()))
