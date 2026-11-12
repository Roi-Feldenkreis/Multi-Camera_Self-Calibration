@echo off
rem ---------------------------------------------------------------------------
rem  Rig test: do steeper camera views (offsets x 0.7, mean viewing angle ~21 deg
rem  instead of ~29 deg) make the seed succeed more often?
rem  Builds two datasets on the steeper rig with the cloud/noise seeds of the two
rem  hardest datasets (s4, s2), writes the new rig's true [CameraGeometry] block,
rem  sets 25 iterations (no divergence guard) and runs both.
rem  Logs (in logs\rig\): rig_log_r07s4.txt, rig_log_r07s2.txt
rem  Needs in the project root: make_steeper_rig.py, write_camera_geometry.py,
rem  make_camera_geometry.py, make_wide_cloud.py, generate_refracted_caldata.py, set_cfg_key.py
rem  Disable sleep first:  powercfg /change standby-timeout-ac 0
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\rig mkdir logs\rig

echo ===== steeper rig =====
%PY% scripts\data_generation\make_steeper_rig.py TestSystem\base\caldata_flipped_step2 TestSystem\rig\caldata_flipped_step2_r07 --factor 0.7

for %%s in (4 2) do (
  echo ===== dataset r07s%%s =====
  %PY% scripts\data_generation\make_wide_cloud.py TestSystem\rig\caldata_flipped_step2_r07 TestSystem\rig\caldata_flipped_step2_r07_s%%s --scale 3 --seed %%s --n 400
  %PY% scripts\data_generation\generate_refracted_caldata.py TestSystem\rig\caldata_flipped_step2_r07_s%%s TestSystem\rig\caldata_refracted_wide400_r07s%%s --noise-seed %%s
  copy /Y TestSystem\base\caldata_refracted_wide400\multicamselfcal.cfg TestSystem\rig\caldata_refracted_wide400_r07s%%s\multicamselfcal.cfg
  %PY% scripts\data_generation\write_camera_geometry.py --gt TestSystem\rig\caldata_refracted_wide400_r07s%%s\ground_truth.json --cfg TestSystem\rig\caldata_refracted_wide400_r07s%%s\multicamselfcal.cfg
)
%PY% scripts\data_generation\set_cfg_key.py Calibration Global-Iteration-Max 25 r07s4 r07s2

echo ===== running =====
copy /Y config_paths.txt config_paths_before_rig.txt
for %%d in (r07s4 r07s2) do (
  echo Running %%d ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\rig\caldata_refracted_wide400_%%d\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\rig\caldata_refracted_wide400_%%d
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\rig\rig_log_%%d.txt 2>&1
)
copy /Y config_paths_before_rig.txt config_paths.txt
del config_paths_before_rig.txt
echo Done: rig_log_r07s4.txt, rig_log_r07s2.txt
