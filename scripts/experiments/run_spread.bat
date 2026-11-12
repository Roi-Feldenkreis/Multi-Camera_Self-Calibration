@echo off
rem ---------------------------------------------------------------------------
rem  Runs the pipeline once on each spread dataset (s2..s5), one log per dataset:
rem  spread_log_s2.txt ... spread_log_s5.txt. Restores config_paths.txt at the end.
rem  Disable sleep first:  powercfg /change standby-timeout-ac 0
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
set PYTHONIOENCODING=utf-8
set ROOT=%CD%
if not exist logs\spread mkdir logs\spread
copy /Y config_paths.txt config_paths_before_spread.txt
for %%s in (2 3 4 5) do (
  echo Running dataset s%%s ...
  > config_paths.txt echo config_file=%ROOT%\TestSystem\spread\caldata_refracted_wide400_s%%s\multicamselfcal.cfg
  >> config_paths.txt echo data_folder=%ROOT%\TestSystem\spread\caldata_refracted_wide400_s%%s
  rmdir /s /q CoreFunctions\__pycache__ Multimedia\__pycache__ OutputFunctions\__pycache__ CommonCfgAndIO\__pycache__ 2>nul
  %PY% GoCal_derefract.py > logs\spread\spread_log_s%%s.txt 2>&1
)
copy /Y config_paths_before_spread.txt config_paths.txt
del config_paths_before_spread.txt
echo All four runs finished: spread_log_s2.txt ... spread_log_s5.txt
