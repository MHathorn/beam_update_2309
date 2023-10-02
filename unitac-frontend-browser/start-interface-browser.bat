REM Activate environment
set MAMBAPATH=%USERPROFILE%\mambaforge
call "%MAMBAPATH%\Scripts\activate.bat" beam

echo %cd%
start "frontend" unitac-frontend-win_x64.exe 
python ../unitac-backend/init.py 
python ../unitac-backend/main.py 

if [ $? -eq 0 ]; then
    echo OK
else
    echo FAIL
fi

REM Deactivate the virtual environment
call "%MAMBAPATH%\Scripts\deactivate.bat"

exit 0