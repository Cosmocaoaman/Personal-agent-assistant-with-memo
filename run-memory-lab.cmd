@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run start.cmd once to prepare the project virtual environment.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "from importlib.metadata import version; assert version('langgraph') == '1.2.14' and version('numpy') == '2.2.6' and version('httpx') == '0.28.1'" >nul 2>nul
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
echo FICTIONAL TEST DATA - separate database, port 8767.
.venv\Scripts\python.exe tools\memory_lab.py --serve --open
if errorlevel 1 pause
