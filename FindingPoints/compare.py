import numpy as np, sys
pos=np.load('t/pos.npy'); refl=np.load('t/refl.npy'); M,_,N=pos.shape
vis=~np.isnan(pos[:,0])
def ev(d):
    I=np.loadtxt(f'{d}/IdMat.dat'); P=np.loadtxt(f'{d}/Points.dat')
    r={}
    det=I==1; e=np.full((M,N),np.nan)
    for i in range(M): e[i]=np.hypot(P[3*i]-1-pos[i,0],P[3*i+1]-1-pos[i,1])
    ok=det&vis&~refl; good=e[ok]
    r['detected']=int(det.sum()); r['clean visible']=int((vis&~refl).sum())
    r['missed clean']=int((vis&~refl&~det).sum())
    r['false (no LED)']=int((det&~vis).sum())
    r['refl frames accepted']=int((det&refl).sum()); r['  of which wrong (>2px)']=int((det&refl&(e>2)).sum())
    r['RMS px']=round(float(np.sqrt(np.mean(good**2))),4); r['max px']=round(float(good.max()),3)
    r['>=3 cams']=int((I.sum(0)>=3).sum())
    return r,P,I
names=sys.argv[1:]; res=[ev(d) for d in names]
keys=list(res[0][0])
print(f"{'metric':26s}"+"".join(f"{n.split('/')[-1]:>14s}" for n in names))
for k in keys: print(f"{k:26s}"+"".join(f"{str(r[0][k]):>14s}" for r in res))
if len(res)>=2:
    P1,I1=res[0][1],res[0][2]; P2,I2=res[1][1],res[1][2]
    both=(I1==1)&(I2==1); d=[]
    for i in range(M):
        m=both[i]; d.append(np.hypot(P1[3*i,m]-P2[3*i,m],P1[3*i+1,m]-P2[3*i+1,m]))
    d=np.concatenate(d); print(f"\n{names[0]} vs {names[1]}: {both.sum()} common points, mean |diff| {d.mean():.3f} px, max {d.max():.3f} px, agree on IdMat {100*(I1==I2).mean():.1f}%")
