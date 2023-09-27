REM Activate environment
set MAMBAPATH=%USERPROFILE%\mambaforge
call "%MAMBAPATH%\Scripts\activate.bat" beam_test

echo %cd%
python ./unitac-backend/updates.py 

if [ $? -eq 0 ]; then
    echo OK
else
    echo FAIL
fi

REM Deactivate the virtual environment
call "%MAMBAPATH%\Scripts\deactivate.bat"

exit 0