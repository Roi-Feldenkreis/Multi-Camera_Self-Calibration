@echo off
rem ---------------------------------------------------------------------------
rem  Intrinsics test 2 (s1, realistic inputs, Intrinsics: estimate, LOS constraint 5 mm),
rem  after the unit fix in CalibrationSystem.py:
rem    s1_est   : focal-length start from the seed (now in correct pixel units)
rem    s1_estfs : focal-length start from a "datasheet" value, Focal-Start-Px 640
rem               (one value for all cameras, 4-7% above their true 597-616 px)
rem  Logs (in logs\intrinsics\): intr2_log_s1_est.txt, intr2_log_s1_estfs.txt
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\intrinsics mkdir logs\intrinsics
xcopy /E /I /Y /Q TestSystem\intrinsics\caldata_refracted_wide400_s1_est TestSystem\intrinsics\caldata_refracted_wide400_s1_estfs
%PY% scripts\data_generation\set_cfg_key.py Multimedia Focal-Start-Px 640 s1_estfs
copy /Y config_paths.txt config_paths_before_intr.txt
for %%d in (s1_estfs s1_est) do (
  echo Running %%d ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\intrinsics\caldata_refracted_wide400_%%d\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\intrinsics\caldata_refracted_wide400_%%d
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\intrinsics\intr2_log_%%d.txt 2>&1
)
copy /Y config_paths_before_intr.txt config_paths.txt
del config_paths_before_intr.txt
echo Done: intr2_log_s1_estfs.txt, intr2_log_s1_est.txt
