@echo off
SET mypath=%~dp0
call C:\anaconda3\Scripts\activate.bat C:\anaconda3
call C:\anaconda3\Scripts\activate.bat treadmill
python.exe "%mypath%\TreadmillApp.py"