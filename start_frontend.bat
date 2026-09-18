@echo off
setlocal
title Aligner - Frontend (Vite dev server)

cd /d "%~dp0frontend"

REM REFUSE TO RUN FROM THE EXPORT SNAPSHOT. See the note in start_backend.bat:
REM the mirror is a full tree copy with the model checkpoints stripped, so the
REM app starts and looks entirely normal while the AI is silently absent.
echo "%CD%" | findstr /i "Aligner_App_AI_Export" >nul
if not errorlevel 1 goto :wrong_copy

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


goto :eof

:wrong_copy
echo.
echo ==================================================================
echo   WRONG FOLDER - this is the export snapshot, not the project.
echo ==================================================================
echo.
echo   You are in:
echo     %CD%
echo.
echo   Aligner_App_AI_Export is a READING COPY for handing to someone
echo   else. Run this from the project root instead.
echo.
pause
exit /b 1
