@echo off
rem ---------------------------------------------------------------------------
rem  Realism rerun with the STAGED distance weight, on the existing s3_real / s5_real
rem  folders (auto height, noisy distances and line of sight already set):
rem    Dist-Weight-Final: 0.5 px/mm  (handover joint BAs and polish; search stays at 40)
rem    Dist-Misfit-Max:   8 mm       (about twice the distance-measurement noise)
rem  Logs (in logs\realism\): realism2_log_s3.txt, realism2_log_s5.txt
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\realism mkdir logs\realism
%PY% scripts\data_generation\set_cfg_key.py Multimedia Dist-Weight-Final 0.5 s3_real s5_real
%PY% scripts\data_generation\set_cfg_key.py Multimedia Dist-Misfit-Max 8 s3_real s5_real
copy /Y config_paths.txt config_paths_before_realism.txt
for %%s in (5 3) do (
  echo Running s%%s_real ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\realism\caldata_refracted_wide400_s%%s_real\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\realism\caldata_refracted_wide400_s%%s_real
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\realism\realism2_log_s%%s.txt 2>&1
)
copy /Y config_paths_before_realism.txt config_paths.txt
del config_paths_before_realism.txt
echo Done: realism2_log_s5.txt, realism2_log_s3.txt
