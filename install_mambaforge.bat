REM source https://gist.github.com/nimaid/a7d6d793f2eba4020135208a57f5c532
@echo off

set ORIGDIR="%CD%"

set MAMBAPATH=%USERPROFILE%\mambaforge
set MAMBAEXE=%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%-mambainstall.exe
set "OS="
set "MCLINK="

where mamba >nul 2>nul
if %ERRORLEVEL% EQU 0 goto MAMBAFOUND

:INSTALLMAMBA
set MCLINK=https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe

echo Downloading Mamba (This will take a while, please wait)...
powershell -Command "(New-Object Net.WebClient).DownloadFile('%MCLINK%', '%MAMBAEXE%')" >nul 2>nul
if errorlevel 1 goto MAMBAERROR

echo Installing Mamba (This will also take a while, please wait)...
start /wait /min "Installing Mambaforge..." "%MAMBAEXE%" /InstallationType=JustMe /S /D="%MAMBAPATH%"
del "%MAMBAEXE%"
if not exist "%MAMBAPATH%\" (goto MAMBAERROR)

"%MAMBAPATH%\Scripts\mamba.exe" init
if errorlevel 1 goto MAMBAERROR

echo Mamba has been installed!
goto CREATEENV

:MAMBAERROR
echo Mamba install failed!
exit /B 1

:MAMBAFOUND
echo Mamba is already installed!
goto CREATEENV

:CREATEENV echo Creating beam environment from environment.yml (This may take some time, please wait)… 
"%MAMBAPATH%\Scripts\mamba.exe" env create -f beam_test.yml 
if errorlevel 1 goto ENVERROR

echo Beam environment has been created! 
goto END

:ENVERROR
echo Beam environment creation failed!
exit /B 1

:END
exit /B 0
