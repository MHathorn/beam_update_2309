REM source https://gist.github.com/nimaid/a7d6d793f2eba4020135208a57f5c532
@echo off

set ORIGDIR="%CD%"

set MINICONDAPATH=%USERPROFILE%\Miniconda3
set CONDAEXE=%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%-condainstall.exe
set "OS="
set "MCLINK="

where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 goto CONDAFOUND

:INSTALLCONDA
reg Query "HKLM\Hardware\Description\System\CentralProcessor\0" | find /i "x86" > NUL && set OS=32BIT || set OS=64BIT
if %OS%==32BIT set MCLINK=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe
if %OS%==64BIT set MCLINK=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

echo Downloading Miniconda3 (This will take a while, please wait)...
powershell -Command "(New-Object Net.WebClient).DownloadFile('%MCLINK%', '%CONDAEXE%')" >nul 2>nul
if errorlevel 1 goto CONDAERROR

echo Installing Miniconda3 (This will also take a while, please wait)...
start /wait /min "Installing Miniconda3..." "%CONDAEXE%" /InstallationType=JustMe /S /D="%MINICONDAPATH%"
del "%CONDAEXE%"
if not exist "%MINICONDAPATH%\" (goto CONDAERROR)

"%MINICONDAPATH%\Scripts\conda.exe" init
if errorlevel 1 goto CONDAERROR

echo Miniconda3 has been installed!
goto END

:CONDAERROR
echo Miniconda3 install failed!
exit /B 1

:CONDAFOUND
echo Conda is already installed!
goto END


:END
exit /B 0

REM write a batch file to create a conda environment and install cudatoolkit and fastai with conda
REM Path: install_fastai.bat

@echo off

set ORIGDIR="%CD%"
set ENVNAME=fastai
set CONDAENV=%USERPROFILE%\Miniconda3\envs\%ENVNAME%
set CONDAEXE=%USERPROFILE%\Miniconda3\Scripts\conda.exe

echo Creating conda environment %ENVNAME%...
"%CONDAEXE%" create -y -n %ENVNAME% python=3.7
if errorlevel 1 goto CONDAERROR

echo Installing cudatoolkit...
"%CONDAEXE%" install -y -n %ENVNAME% -c pytorch -c fastai fastai
if errorlevel 1 goto CONDAERROR

echo Installing fastai...
"%CONDAEXE%" install -y -n %ENVNAME% -c pytorch -c fastai fastai
if errorlevel 1 goto CONDAERROR

echo Activating conda environment %ENVNAME%...
call "%CONDAEXE%" activate %ENVNAME%
if errorlevel 1 goto CONDAERROR

echo Installing fastai...
pip install fastai
if errorlevel 1 goto CONDAERROR

echo fastai has been installed!
goto END

:CONDAERROR
echo fastai install failed!
exit /B 1

:END
exit /B 0

