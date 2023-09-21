REM Activate pytorch_env
set MINICONDAPATH=%USERPROFILE%\Miniconda3
call "%MINICONDAPATH%\Scripts\activate.bat" testenv

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
call "%MINICONDAPATH%\Scripts\deactivate.bat"

exit 0