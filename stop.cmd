@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Run start.cmd first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe server.py --stop
if errorlevel 1 pause
