@echo off
REM Ensure script runs from the correct location
pushd "%~dp0"

REM Run the Python script to print system environment variables
C:/Users/elsensoy/AppData/Local/Microsoft/WindowsApps/python3.11.exe ^
   c:\Users\elsensoy\.vscode\extensions\ms-python.python-2025.0.0-win32-x64\python_files\printEnvVariablesToFile.py ^
   c:\Users\elsensoy\.vscode\extensions\ms-python.python-2025.0.0-win32-x64\python_files\deactivate\powershell\envVars.txt

REM Set environment variables
set PATH=%PATH%;C:\Users\elsensoy\model_error\model_error_uq_plasma
set PYTHONPATH=C:\Users\elsensoy\model_error\model_error_uq_plasma;%PYTHONPATH%

REM Ensure HallThruster module is available
set HALLTHRUSTER_PATH=C:\Users\elsensoy\.julia\packages\HallThruster\yxE62\python
set PYTHONPATH=%HALLTHRUSTER_PATH%;%PYTHONPATH%

REM Change to hall_opt directory
cd /d "%~dp0\hall_opt"

REM Run main.py
python main.py %*

REM Restore previous directory
popd
