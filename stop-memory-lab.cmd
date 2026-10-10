@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe server.py --port 8767 --data-dir artifacts\memory-lab\sandbox --stop
if errorlevel 1 pause
