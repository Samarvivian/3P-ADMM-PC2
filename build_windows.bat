@echo off
REM 3P-ADMM-PC2 Windows Build Script
REM Output: dist\3P-ADMM-PC2.exe
setlocal enabledelayedexpansion

echo === 3P-ADMM-PC2 Build Script ===
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add to PATH.
    pause & exit /b 1
)
python --version

echo.
echo Installing all dependencies...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause & exit /b 1
)
echo Dependencies OK.

echo.
echo Cleaning previous build...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo.
echo Building executable...
python -m PyInstaller 3P-ADMM-PC2-GUI.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller failed. See output above.
    pause & exit /b 1
)

echo.
if exist "dist\3P-ADMM-PC2.exe" (
    for %%F in ("dist\3P-ADMM-PC2.exe") do set /a SIZE_MB=%%~zF / 1048576
    echo Build SUCCESSFUL
    echo Output : dist\3P-ADMM-PC2.exe
    echo Size   : !SIZE_MB! MB
) else (
    echo ERROR: dist\3P-ADMM-PC2.exe not found after build.
    pause & exit /b 1
)

echo.
pause
