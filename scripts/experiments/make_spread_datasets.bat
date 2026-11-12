@echo off
rem ---------------------------------------------------------------------------
rem  Builds 4 wide-cloud datasets for the accuracy-spread study.
rem  Same cameras; each dataset has its own particle cloud AND its own image-noise
rem  draw (seed s = 2..5). The config is copied from caldata_refracted_wide400, so
rem  it keeps BA-Mode, the Seed- lines, BA-All-Particles and the LUT settings.
rem ---------------------------------------------------------------------------
cd /d C:\Users\User\PycharmProjects\Muticameracalb
set PY=C:\Users\User\anaconda3\python.exe
for %%s in (2 3 4 5) do (
  echo.
  echo ===================== dataset s%%s =====================
  %PY% scripts\data_generation\make_wide_cloud.py TestSystem\base\caldata_flipped_step2 TestSystem\spread\caldata_flipped_step2_wide400_s%%s --scale 3 --seed %%s --n 400
  %PY% scripts\data_generation\generate_refracted_caldata.py TestSystem\spread\caldata_flipped_step2_wide400_s%%s TestSystem\spread\caldata_refracted_wide400_s%%s --noise-seed %%s
  copy /Y TestSystem\base\caldata_refracted_wide400\multicamselfcal.cfg TestSystem\spread\caldata_refracted_wide400_s%%s\multicamselfcal.cfg
)
echo.
echo Done. Check each dataset above: "largest radial distance" must be below 850 mm
echo and the deepest particle below 450 mm (the copied LUT settings).
