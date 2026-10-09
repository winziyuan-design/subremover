import sys
sys.argv=['pipe.py','--frames','0:1']
src=open('pipe.py').read(); src=src[:src.index('sess_o=')]
exec(src)
tot=0
for rname,mf in REG.items():
    ch=0; cur=[]
    for i in range(N):
        if not mf(i).any(): 
            if cur: ch+=1; cur=[]
            continue
        if cur and (shot[i]!=shot[cur[-1]] or len(cur)==10): ch+=1; cur=[]
        cur.append(i)
    if cur: ch+=1
    print(rname,'chunks',ch); tot+=ch
print('N',N,'total calls (before piece split)',tot)
