@echo off
rem ---------------------------------------------------------------------------
rem  Cause test: is a dataset's seed success driven by its particle CLOUD or by its
rem  image NOISE draw? Reuses the existing Step 2 folders, so the clouds are exactly
rem  those of s1 (easy) and s4 (hard); only the noise changes.
rem    c1n4   : s1 cloud + s4 noise draw (seed 4)
rem    c4n123 : s4 cloud + s1 noise draw (seed 123)
rem    c4n0   : s4 cloud, no noise
rem  Config copied from caldata_refracted_wide400 (no divergence guard), 25 iterations.
rem  Logs (in logs\cause\): cause_log_c1n4.txt, cause_log_c4n123.txt, cause_log_c4n0.txt
rem  Disable sleep first:  powercfg /change standby-timeout-ac 0
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\cause mkdir logs\cause

echo ===== building the three test datasets =====
%PY% scripts\data_generation\generate_refracted_caldata.py TestSystem\base\caldata_flipped_step2_wide400 TestSystem\cause\caldata_refracted_wide400_c1n4 --noise-seed 4
%PY% scripts\data_generation\generate_refracted_caldata.py TestSystem\spread\caldata_flipped_step2_wide400_s4 TestSystem\cause\caldata_refracted_wide400_c4n123 --noise-seed 123
%PY% scripts\data_generation\generate_refracted_caldata.py TestSystem\spread\caldata_flipped_step2_wide400_s4 TestSystem\cause\caldata_refracted_wide400_c4n0 --noise-seed 4 --noise-std 0
for %%d in (c1n4 c4n123 c4n0) do copy /Y TestSystem\base\caldata_refracted_wide400\multicamselfcal.cfg TestSystem\cause\caldata_refracted_wide400_%%d\multicamselfcal.cfg
%PY% scripts\data_generation\set_cfg_key.py Calibration Global-Iteration-Max 25 c1n4 c4n123 c4n0

echo ===== running =====
copy /Y config_paths.txt config_paths_before_cause.txt
for %%d in (c1n4 c4n123 c4n0) do (
  echo Running %%d ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\cause\caldata_refracted_wide400_%%d\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\cause\caldata_refracted_wide400_%%d
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\cause\cause_log_%%d.txt 2>&1
)
copy /Y config_paths_before_cause.txt config_paths.txt
del config_paths_before_cause.txt
echo Done: cause_log_c1n4.txt, cause_log_c4n123.txt, cause_log_c4n0.txt
