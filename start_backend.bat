@echo off
setlocal
title Aligner - Backend (api_core:app)

REM Run from the repo root no matter where this was launched from.
cd /d "%~dp0"

REM REFUSE TO RUN FROM THE EXPORT SNAPSHOT.
REM
REM Aligner_App_AI_Export\ is a reading copy. build_ai_export.py strips every
REM model checkpoint from it by design, but it is a full tree copy otherwise -
REM so the app STARTS, the viewport WORKS, every recent fix is present, and the
REM only thing missing is the AI. There is nothing to notice, which is exactly
REM why this has to be a refusal and not a warning.
echo "%CD%" | findstr /i "Aligner_App_AI_Export" >nul
if not errorlevel 1 goto :wrong_copy

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo ==================================================================
echo   PROJECT ALIGNER  -  BACKEND   (api_core:app)
echo ==================================================================
echo   Interpreter : %PY%
echo   Directory   : %CD%
echo.

echo [1/3] Required packages...
"%PY%" -c "import fastapi, uvicorn; print('      fastapi', fastapi.__version__, '| uvicorn', uvicorn.__version__)"
if errorlevel 1 goto :missing_deps

REM python-multipart is imported as python_multipart on 0.0.12+ and as
REM multipart before that. Accept either; FastAPI needs one of them to read
REM the uploaded STL, and without it every upload fails with an opaque 500.
"%PY%" -c "import python_multipart" 2>nul
if errorlevel 1 "%PY%" -c "import multipart" 2>nul
if errorlevel 1 goto :missing_deps
echo       python-multipart OK

echo [2/3] AI checkpoint...
if exist "ToothGroupNetwork\ckpts\0707_cosannealing_val.h5" (
    echo       found  ToothGroupNetwork\ckpts\0707_cosannealing_val.h5
) else (
    echo       NOT FOUND - the API still starts and manual cutting still works,
    echo       but "Segment Teeth" will report the model as unavailable.
)

echo [3/3] Port 8000...
netstat -ano | findstr "LISTENING" | findstr ":8000" >nul
if not errorlevel 1 goto :port_busy
echo       free
echo.
echo ------------------------------------------------------------------
echo  Starting uvicorn on http://127.0.0.1:8000
echo.
echo  The API accepts requests IMMEDIATELY. The segmentation model keeps
echo  loading in a background thread - watch for "[Clinical AI] Pipeline
echo  ready" below, or poll http://127.0.0.1:8000/api/ai/status
echo.
echo  Leave this window open. Ctrl+C stops the server.
echo ------------------------------------------------------------------
echo.

"%PY%" -m uvicorn api_core:app --host 127.0.0.1 --port 8000 --reload

echo.
echo Server stopped.
pause
exit /b 0


:missing_deps
echo.
echo   MISSING DEPENDENCY - the API cannot start without these.
echo   Install them with:
echo.
echo       "%PY%" -m pip install fastapi uvicorn python-multipart
echo.
pause
exit /b 1


:port_busy
echo.
echo   PORT 8000 IS ALREADY IN USE. Starting a second uvicorn here would die
echo   with a bare [WinError 10048] and look like a broken backend.
echo   The process holding it:
echo.
netstat -ano | findstr "LISTENING" | findstr ":8000"
echo.
echo   Either that server is already serving the app - just reload the
echo   browser - or stop it with:   taskkill /PID ^<pid^> /F
echo.
pause
exit /b 1


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
echo   else. It carries no AI model checkpoints by design, so the app
echo   would start, look completely normal, and report the AI as
echo   unavailable with no hint as to why.
echo.
echo   Go up one folder and run this from the project root instead:
echo     ..\%~nx0
echo.
pause
exit /b 1
