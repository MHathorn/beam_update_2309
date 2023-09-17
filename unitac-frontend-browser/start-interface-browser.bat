echo %cd%
start "frontend" unitac-frontend-win_x64.exe 
python ../unitac-backend/init.py 
python ../unitac-backend/main.py 

if [ $? -eq 0 ]; then
    echo OK
else
    echo FAIL
fi

exit 0

