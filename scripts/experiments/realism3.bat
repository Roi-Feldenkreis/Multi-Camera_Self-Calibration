@echo off
rem ---------------------------------------------------------------------------
rem  Realism runs on two EASY datasets, to learn from successes too:
rem    s1_real    : copy of caldata_refracted_wide400        (original rig, s1)
rem    r07s2_real : copy of caldata_refracted_wide400_r07s2  (steeper rig)
rem  Realistic inputs + frozen realism settings:
rem    Camera-Glass-Distance: auto | distances +-4 mm, line of sight +-5 mm (noise
rem    seeds 1 and 2) | Dist-Weight-Final 0.5 | Dist-Misfit-Max 8 | guard off | 25 iterations
rem  Logs (in logs\realism\): realism3_log_s1.txt, realism3_log_r07s2.txt
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\realism mkdir logs\realism

echo ===== preparing copies =====
xcopy /E /I /Y /Q TestSystem\base\caldata_refracted_wide400 TestSystem\realism\caldata_refracted_wide400_s1_real
xcopy /E /I /Y /Q TestSystem\rig\caldata_refracted_wide400_r07s2 TestSystem\realism\caldata_refracted_wide400_r07s2_real
%PY% scripts\data_generation\write_camera_geometry.py --gt TestSystem\realism\caldata_refracted_wide400_s1_real\ground_truth.json --cfg TestSystem\realism\caldata_refracted_wide400_s1_real\multicamselfcal.cfg --noisy --seed 1
%PY% scripts\data_generation\write_camera_geometry.py --gt TestSystem\realism\caldata_refracted_wide400_r07s2_real\ground_truth.json --cfg TestSystem\realism\caldata_refracted_wide400_r07s2_real\multicamselfcal.cfg --noisy --seed 2
%PY% scripts\data_generation\set_cfg_key.py Multimedia Camera-Glass-Distance auto s1_real r07s2_real
%PY% scripts\data_generation\set_cfg_key.py Multimedia Dist-Weight-Final 0.5 s1_real r07s2_real
%PY% scripts\data_generation\set_cfg_key.py Multimedia Dist-Misfit-Max 8 s1_real r07s2_real
%PY% scripts\data_generation\set_cfg_key.py Multimedia Derefract-Max-Shift-Px 0 s1_real r07s2_real
%PY% scripts\data_generation\set_cfg_key.py Calibration Global-Iteration-Max 25 s1_real r07s2_real

echo ===== running =====
copy /Y config_paths.txt config_paths_before_realism.txt
for %%d in (s1_real r07s2_real) do (
  echo Running %%d ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\realism\caldata_refracted_wide400_%%d\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\realism\caldata_refracted_wide400_%%d
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\realism\realism3_log_%%d.txt 2>&1
)
copy /Y config_paths_before_realism.txt config_paths.txt
del config_paths_before_realism.txt
echo Done: realism3_log_s1_real.txt, realism3_log_r07s2_real.txt
