@echo off
rem ---------------------------------------------------------------------------
rem  Realism run: only inputs a real experiment can supply.
rem    - Camera-Glass-Distance: auto   (derived from the line-of-sight distances)
rem    - inter-camera distances with 4 mm hand-measurement noise
rem    - line-of-sight distances with 5 mm hand-measurement noise
rem    - frozen settings: handover limit 1 px (default), divergence guard off, 25 iterations
rem  Works on COPIES of s3 and s5 (folders ..._s3_real, ..._s5_real); the originals
rem  and their baselines stay untouched.
rem  Logs (in logs\realism\): realism_log_s3.txt, realism_log_s5.txt
rem  Needs in the project root: write_camera_geometry.py, make_camera_geometry.py, set_cfg_key.py
rem  Disable sleep first:  powercfg /change standby-timeout-ac 0
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\realism mkdir logs\realism

for %%s in (3 5) do (
  echo ===== preparing s%%s_real =====
  xcopy /E /I /Y /Q TestSystem\spread\caldata_refracted_wide400_s%%s TestSystem\realism\caldata_refracted_wide400_s%%s_real
  %PY% scripts\data_generation\write_camera_geometry.py --gt TestSystem\realism\caldata_refracted_wide400_s%%s_real\ground_truth.json --cfg TestSystem\realism\caldata_refracted_wide400_s%%s_real\multicamselfcal.cfg --noisy --seed %%s
)
%PY% scripts\data_generation\set_cfg_key.py Multimedia Camera-Glass-Distance auto s3_real s5_real
%PY% scripts\data_generation\set_cfg_key.py Multimedia Derefract-Max-Shift-Px 0 s3_real s5_real
%PY% scripts\data_generation\set_cfg_key.py Calibration Global-Iteration-Max 25 s3_real s5_real

echo ===== running =====
copy /Y config_paths.txt config_paths_before_realism.txt
for %%s in (3 5) do (
  echo Running s%%s_real ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\realism\caldata_refracted_wide400_s%%s_real\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\realism\caldata_refracted_wide400_s%%s_real
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\realism\realism_log_s%%s.txt 2>&1
)
copy /Y config_paths_before_realism.txt config_paths.txt
del config_paths_before_realism.txt
echo Done: realism_log_s3.txt, realism_log_s5.txt
