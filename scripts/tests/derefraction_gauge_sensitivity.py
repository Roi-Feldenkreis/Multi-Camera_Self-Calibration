import sys; sys.path.insert(0,'.')
import numpy as np
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root on path
from Multimedia.Multimedia_params import MultimediaParams, CameraParams
from Multimedia.Iterative_radial_shift import multimedia_radial_shift

rng = np.random.default_rng(7)
def look_at(C, t, up=np.array([0,0,1.0])):
    z=t-C; z/=np.linalg.norm(z); x=np.cross(up,z)
    if np.linalg.norm(x)<1e-6: x=np.cross(np.array([0,1.,0]),z)
    x/=np.linalg.norm(x); y=np.cross(z,x); return np.array([x,y,z])

mm=MultimediaParams(n1=1.0,n2=[1.46],d=[5.0],n3=1.33)
C=np.array([[-120,-120,460.],[120,-120,455.],[120,120,450.],[-120,120,465.]]); NCAM=4
fx=600.; W,H=752,480; K=np.array([[fx,0,W/2],[0,fx,H/2],[0,0,1.]])
M=200; cloud=np.column_stack([rng.uniform(-70,70,M),rng.uniform(-70,70,M),rng.uniform(-150,-30,M)])
cams=[CameraParams(cam_id=k,position=C[k],angles=np.zeros(3),c=fx,mm=mm) for k in range(NCAM)]
R=[look_at(C[k],cloud.mean(0)) for k in range(NCAM)]
Pe=np.zeros((3*NCAM,4))
for k in range(NCAM):
    Pe[3*k:3*k+3,:3]=K@R[k]; Pe[3*k:3*k+3,3]=K@(-R[k]@C[k])
def proj(Pk,X3):
    Xh=np.vstack([X3,np.ones(X3.shape[1])]); x=Pk@Xh; return x[:2]/x[2]

# synthesize refracted obs with TRUE geometry (noise-free to isolate gauge effect)
Ws=np.full((3*NCAM,M),np.nan); Id=np.zeros((NCAM,M),int)
for k in range(NCAM):
    for j in range(M):
        Pbar,_,_=multimedia_radial_shift(cloud[j],cams[k],eps=1e-7,max_iter=300,ocf=1.3)
        xr=proj(Pe[3*k:3*k+3],Pbar.reshape(3,1)).ravel()
        if 0<=xr[0]<=W and 0<=xr[1]<=H:
            Ws[3*k:3*k+2,j]=xr; Ws[3*k+2,j]=1; Id[k,j]=1

def triangulate(Ws):
    X=np.full((3,M),np.nan)
    for j in range(M):
        rows=[]
        for k in range(NCAM):
            if Id[k,j]:
                u,v=Ws[3*k,j],Ws[3*k+1,j]; Pk=Pe[3*k:3*k+3]
                rows+=[u*Pk[2]-Pk[0], v*Pk[2]-Pk[1]]
        if len(rows)>=4:
            _,_,Vt=np.linalg.svd(np.array(rows)); X[:,j]=Vt[-1][:3]/Vt[-1][3]
    return X
def rms(X): m=~np.isnan(X[0]); return float(np.sqrt(np.nanmean(np.sum((X[:,m]-cloud[m].T)**2,0))))

# de-refraction loop where the bootstrap BELIEVES the scale is g x truth
def run(g):
    cams_g=[CameraParams(cam_id=k,position=C[k]*g,angles=np.zeros(3),c=fx,mm=mm) for k in range(NCAM)]
    Ww=Ws.copy(); prev=None
    for it in range(1,16):
        Xe=triangulate(Ww)
        delta=np.zeros((3*NCAM,M))
        for k in range(NCAM):
            for j in range(M):
                if Id[k,j] and not np.isnan(Xe[0,j]):
                    # shift computed in the g-scaled world (t stays 5mm), then unscaled
                    Pbar_s,_,_=multimedia_radial_shift(Xe[:,j]*g,cams_g[k],eps=1e-7,max_iter=300,ocf=1.3)
                    Pbar=Pbar_s/g
                    xp=proj(Pe[3*k:3*k+3],Xe[:,j:j+1]).ravel()
                    xr=proj(Pe[3*k:3*k+3],Pbar.reshape(3,1)).ravel()
                    delta[3*k:3*k+2,j]=xr-xp
        Ww=Ws-delta
        chg=np.inf if prev is None else float(np.nanmax(np.abs(delta-prev)))
        if chg<1e-3: break
        prev=delta
    return rms(triangulate(Ww)), it

print("gauge g = bootstrap_scale / true_scale  (g=1.0 is exact)")
print(f"{'g':>6} {'final cloud RMS (mm)':>22} {'passes':>8}")
for g in [0.7,0.8,0.9,1.0,1.1,1.25,1.5]:
    r,it=run(g); print(f"{g:>6.2f} {r:>22.4f} {it:>8d}")
