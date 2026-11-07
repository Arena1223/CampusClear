@echo off
setlocal
set "HERE=%~dp0"

if defined CAMPUSCLEAR_PYLIBS (
    set "PYLIBS=%CAMPUSCLEAR_PYLIBS%"
) else if exist "%HERE%pylibs\fastapi" (
    set "PYLIBS=%HERE%pylibs"
) else if exist "I:\MOKOUSPACE\PROGRAM\CampusClear\pylibs\fastapi" (
    set "PYLIBS=I:\MOKOUSPACE\PROGRAM\CampusClear\pylibs"
) else (
    set "PYLIBS=%HERE%pylibs"
)

set "PYTHONPATH=%HERE%src;%PYLIBS%"
python "%HERE%tools\diagnose_server.py" %*
exit /b %errorlevel%
