@echo off
setlocal
title Aligner - Frontend (Vite dev server)

cd /d "%~dp0frontend"

echo ==================================================================
echo   PROJECT ALIGNER  -  FRONTEND   (Vite dev server)
echo ==================================================================
echo   Directory : %CD%
echo.

where npm >nul 2>&1
if errorlevel 1 (
    echo   npm was not found on PATH. Install Node.js ^(which ships npm^)
    echo   from https://nodejs.org and reopen this window.
    echo.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo [1/2] node_modules missing - running npm install ^(one time, slow^)...
    call npm install
    if errorlevel 1 (
        echo.
        echo   npm install failed. Fix the error above, then rerun this script.
        echo.
        pause
        exit /b 1
    )
) else (
    echo [1/2] node_modules present - skipping npm install.
)

echo [2/2] Backend reachability...
curl -s -o nul -m 2 http://127.0.0.1:8000/api/ai/status
if errorlevel 1 (
    echo       http://127.0.0.1:8000 is not answering yet.
    echo       Start start_backend.bat in another window, or the first
    echo       upload will fail with "Backend unavailable".
) else (
    echo       http://127.0.0.1:8000 is up.
)

echo.
echo ------------------------------------------------------------------
echo  Starting Vite. Open the http://localhost:5173 URL it prints below.
echo  Leave this window open. Ctrl+C stops the dev server.
echo ------------------------------------------------------------------
echo.

call npm run dev

echo.
echo Dev server stopped.
pause
