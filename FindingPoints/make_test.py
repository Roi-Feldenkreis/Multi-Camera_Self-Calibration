import numpy as np, cv2, os
rng=np.random.default_rng(1)
M,N,W,H=4,600,640,480
f=700; K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1]])
def look(C):
    z=-C/np.linalg.norm(C); x=np.cross([0,0,1],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.vstack([x,y,z]); return K@np.hstack([R,-R@C[:,None]])
Cs=[np.array([600*np.cos(a),600*np.sin(a),300]) for a in np.linspace(0,2*np.pi,M,endpoint=False)]
P=[look(C) for C in Cs]
X=rng.uniform(-170,170,(N,3))
pos=np.full((M,2,N),np.nan); refl=np.zeros((M,N),bool)
yy,xx=np.mgrid[0:H,0:W]
for i in range(M):
    os.makedirs(f"t/cam{i+1}",exist_ok=True); os.makedirs(f"t/bg{i+1}",exist_ok=True)
    bgimg=20+10*np.sin(xx/50.)+5*np.cos(yy/30.)
    for k in range(60):
        cv2.imwrite(f"t/bg{i+1}/bg_{k:04d}.tif",np.clip(bgimg+rng.normal(0,2,(H,W)),0,255).astype(np.uint8))
    for j in range(N):
        img=bgimg+rng.normal(0,2,(H,W))
        x=P[i]@np.r_[X[j],1]; u,v=x[:2]/x[2]
        if 0<=u<W and 0<=v<H and rng.random()>0.08:
            A=rng.uniform(110,220)
            img+=A*np.exp(-0.5*((xx-u)**2+(yy-v)**2)/1.5**2); pos[i,:,j]=(u,v)
            if rng.random()<0.04:
                img+=A*rng.uniform(0.5,0.9)*np.exp(-0.5*((xx-u-60)**2+(yy-v-20)**2)/1.5**2); refl[i,j]=True
        cv2.imwrite(f"t/cam{i+1}/c{i+1}_{j:06d}.tif",np.clip(img,0,255).astype(np.uint8))
np.save("t/pos.npy",pos); np.save("t/refl.npy",refl)
print("visible:",(~np.isnan(pos[:,0])).sum(1),"reflection frames:",refl.sum(1))
