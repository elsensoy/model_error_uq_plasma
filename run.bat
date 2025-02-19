@echo off
REM Run the pre-commands

REM Run the Python script to print environment variables
C:/Users/elsensoy/AppData/Local/Microsoft/WindowsApps/python3.11.exe ^
   c:\Users\elsensoy\.vscode\extensions\ms-python.python-2025.0.0-win32-x64\python_files\printEnvVariablesToFile.py ^
   c:\Users\elsensoy\.vscode\extensions\ms-python.python-2025.0.0-win32-x64\python_files\deactivate\powershell\envVars.txt

REM Set environment variables
set PATH=%PATH%;C:\Users\elsensoy\model_error\model_error_uq_plasma
set PYTHONPATH=C:\Users\elsensoy\model_error\model_error_uq_plasma;%PYTHONPATH%

cd /d "%~dp0\hall_opt"
python "%~dp0\hall_opt\main.py" %*
