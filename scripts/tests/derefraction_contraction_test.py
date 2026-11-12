"""
derefraction_contraction_test.py
================================
Standalone proof that the Method-B de-refraction fixed point converges.

Plants a 4-camera rig (glass top z=0, cameras above, cloud below), synthesises
REFRACTED observations with the exact forward model (multimedia_radial_shift +
pinhole projection of P_bar), then runs the de-refraction fixed point using the
production LUT and linear triangulation from the KNOWN cameras.

This ISOLATES the de-refraction inversion (known cameras + triangulation); it does
NOT test Svoboda pose estimation under refraction -- that is exercised only in the
live GoCal pipeline. Run:  python derefraction_contraction_test.py [noise_px]

Imports the project's real Multimedia package (Snell_core, Multimedia_params,
Iterative_radial_shift, Multimedia_lut).
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root on path
from Multimedia.Multimedia_params import MultimediaParams, CameraParams
from Multimedia.Iterative_radial_shift import multimedia_radial_shift
from Multimedia.Multimedia_lut import MultimediaLUT, MultiCameraLUT

rng = np.random.default_rng(7)

# ---- geometry: glass top z=0, cameras above, cloud below ----
def look_at(C, target, up=np.array([0,0,1.0])):
    z = target - C; z/=np.linalg.norm(z)
    x = np.cross(up, z); 
    if np.linalg.norm(x)<1e-6: x=np.cross(np.array([0,1.,0]),z)
    x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.array([x,y,z])

mm = MultimediaParams(n1=1.0, n2=[1.46], d=[5.0], n3=1.33)
t = mm.t_total
C = np.array([[-120,-120,460.],[120,-120,455.],[120,120,450.],[-120,120,465.]])
NCAM = 4
fx = 600.0; W,H = 752,480; cx,cy = W/2, H/2
K = np.array([[fx,0,cx],[0,fx,cy],[0,0,1.]])

# cloud below glass bottom (z < -t)
M = 200
cloud = np.column_stack([rng.uniform(-70,70,M), rng.uniform(-70,70,M), rng.uniform(-150,-30,M)])

cams_radial = [CameraParams(cam_id=k, position=C[k], angles=np.zeros(3), c=fx, mm=mm) for k in range(NCAM)]
R = [look_at(C[k], cloud.mean(0)) for k in range(NCAM)]

# true pinhole projection matrices (pixel space): Pe = K[R | -R C]
Pe = np.zeros((3*NCAM,4))
for k in range(NCAM):
    Pe[3*k:3*k+3,:3] = K @ R[k]
    Pe[3*k:3*k+3, 3] = K @ (-R[k] @ C[k])

def proj(Pe_k, X3):  # X3 (3,N) -> (2,N) pixels
    Xh = np.vstack([X3, np.ones(X3.shape[1])])
    x = Pe_k @ Xh; return x[:2]/x[2]

# ---- synthesize: TRUE pinhole obs and REFRACTED obs ----
NOISE = float(sys.argv[1]) if len(sys.argv)>1 else 0.0
Ws_true = np.full((3*NCAM, M), np.nan)   # pinhole obs of true cloud (the target)
Ws_raw  = np.full((3*NCAM, M), np.nan)   # refracted obs (what cameras record)
IdMat   = np.zeros((NCAM, M), int)
for k in range(NCAM):
    Pe_k = Pe[3*k:3*k+3,:]
    for j in range(M):
        P = cloud[j]
        Pbar,_,_ = multimedia_radial_shift(P, cams_radial[k], eps=1e-7, max_iter=300, ocf=1.3)
        # pinhole obs of true P (no refraction) -- the de-refraction target
        xp = proj(Pe_k, P.reshape(3,1)).ravel()
        # refracted obs = pinhole projection of the shifted P_bar
        xr = proj(Pe_k, Pbar.reshape(3,1)).ravel() + rng.normal(0,NOISE,2)
        if 0<=xr[0]<=W and 0<=xr[1]<=H:
            Ws_true[3*k:3*k+2,j]=xp; Ws_true[3*k+2,j]=1
            Ws_raw[3*k:3*k+2,j]=xr;  Ws_raw[3*k+2,j]=1
            IdMat[k,j]=1

# ---- build production LUT for the de-refraction correction ----
mc = MultiCameraLUT(cams_radial, z_glass_top=0.0)
mc.build_all(R_max=200.0, Zp_max=260.0, n_R=160, n_Zp=160, verbose=False)

# ---- linear DLT triangulation from KNOWN cameras (isolates de-refraction) ----
def triangulate(Pe, Ws, IdMat):
    M = Ws.shape[1]; X=np.full((3,M),np.nan)
    for j in range(M):
        rows=[]
        for k in range(NCAM):
            if IdMat[k,j]:
                u,v = Ws[3*k,j], Ws[3*k+1,j]; Pk=Pe[3*k:3*k+3,:]
                rows.append(u*Pk[2]-Pk[0]); rows.append(v*Pk[2]-Pk[1])
        if len(rows)>=4:
            A=np.array(rows); _,_,Vt=np.linalg.svd(A); Xh=Vt[-1]; X[:,j]=Xh[:3]/Xh[3]
    return X

def lut_shift_all(Xe3, IdMat):
    # P_bar per camera via LUT (production path)
    out={}
    for k in range(NCAM):
        Pbar=np.full((3,Xe3.shape[1]),np.nan)
        for j in range(Xe3.shape[1]):
            if IdMat[k,j] and not np.isnan(Xe3[0,j]):
                Pbar[:,j]=mc[k].shift_point(Xe3[:,j], cams_radial[k])
        out[k]=Pbar
    return out

def derefract_delta(Pe, Xe3, IdMat):
    delta=np.zeros((3*NCAM, Xe3.shape[1]))
    shifts=lut_shift_all(Xe3, IdMat)
    for k in range(NCAM):
        Pe_k=Pe[3*k:3*k+3,:]
        for j in range(Xe3.shape[1]):
            if IdMat[k,j] and not np.isnan(Xe3[0,j]):
                xp=proj(Pe_k, Xe3[:,j:j+1]).ravel()
                xr=proj(Pe_k, shifts[k][:,j:j+1]).ravel()
                delta[3*k:3*k+2,j]=xr-xp
    return delta

def cloud_rms(Xe3):
    m=~np.isnan(Xe3[0]); return float(np.sqrt(np.nanmean(np.sum((Xe3[:,m]-cloud[m].T)**2,0))))
def reproj_vs_true(Ws):
    e=[]
    for k in range(NCAM):
        for j in range(M):
            if IdMat[k,j]:
                d=Ws[3*k:3*k+2,j]-Ws_true[3*k:3*k+2,j]; e.append(np.hypot(*d))
    return float(np.mean(e))

print(f"=== de-refraction contraction test (noise={NOISE} px) ===")
print(f"cloud={M} pts, {NCAM} cams; mean refractive pixel shift (raw vs true) = {reproj_vs_true(Ws_raw):.2f} px")
X0=triangulate(Pe, Ws_raw, IdMat)
print(f"iter 0 (no correction): cloud RMS error = {cloud_rms(X0):8.3f} mm   |   obs err vs true = {reproj_vs_true(Ws_raw):7.3f} px")

Ws_work=Ws_raw.copy(); prev=None
for it in range(1,11):
    Xe3=triangulate(Pe, Ws_work, IdMat)
    delta=derefract_delta(Pe, Xe3, IdMat)
    Ws_work = Ws_raw - delta
    # restore homogeneous weight rows (delta has 0 there already)
    chg = np.inf if prev is None else float(np.nanmax(np.abs(delta-prev)))
    print(f"iter {it:2d}: cloud RMS = {cloud_rms(Xe3):8.4f} mm   |   obs err vs true = {reproj_vs_true(Ws_work):7.4f} px   |   max|dDelta| = {chg:8.4f} px")
    if chg<1e-3: 
        Xe3=triangulate(Pe, Ws_work, IdMat)
        print(f"  converged. final cloud RMS = {cloud_rms(Xe3):.4f} mm"); break
    prev=delta
