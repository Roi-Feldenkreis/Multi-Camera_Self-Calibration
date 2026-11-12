@echo off
rem ---------------------------------------------------------------------------
rem  Intrinsics test on s1 with REALISTIC inputs (copies of caldata_refracted_wide400_s1_real:
rem  auto height, noisy distances and line of sight, staged distance weight):
rem    s1_radlos : Intrinsics rad      + line of sight as weak constraint (LOS-Sigma-Final-Mm 5)
rem    s1_est    : Intrinsics estimate + line of sight as weak constraint (no .rad values used)
rem  Container predictions (optimum of these inputs):
rem    s1_radlos : tilt ~0.14 deg, glass frame ~4.6 mm, shape ~1.2 mm
rem    s1_est    : tilt ~0.85 deg, glass frame ~10.4 mm, shape ~1.5 mm
rem  Logs (in logs\intrinsics\): intr_log_s1_radlos.txt, intr_log_s1_est.txt
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\intrinsics mkdir logs\intrinsics
for %%d in (s1_radlos s1_est) do xcopy /E /I /Y /Q TestSystem\realism\caldata_refracted_wide400_s1_real TestSystem\intrinsics\caldata_refracted_wide400_%%d
%PY% scripts\data_generation\set_cfg_key.py Multimedia LOS-Sigma-Final-Mm 5 s1_radlos s1_est
%PY% scripts\data_generation\set_cfg_key.py Multimedia Intrinsics rad s1_radlos
%PY% scripts\data_generation\set_cfg_key.py Multimedia Intrinsics estimate s1_est
copy /Y config_paths.txt config_paths_before_intr.txt
for %%d in (s1_radlos s1_est) do (
  echo Running %%d ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\intrinsics\caldata_refracted_wide400_%%d\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\intrinsics\caldata_refracted_wide400_%%d
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\intrinsics\intr_log_%%d.txt 2>&1
)
copy /Y config_paths_before_intr.txt config_paths.txt
del config_paths_before_intr.txt
echo Done: intr_log_s1_radlos.txt, intr_log_s1_est.txt
