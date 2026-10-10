@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -c "from importlib.metadata import version; assert version('langgraph') == '1.2.14' and version('numpy') == '2.2.6' and version('httpx') == '0.28.1'" >nul 2>nul
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe server.py --open
if errorlevel 1 pause
