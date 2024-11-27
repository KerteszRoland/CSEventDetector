@echo off
nvcc mask.cu -o mask.exe -lgdi32 -luser32 -O2 -ldwmapi
if %ERRORLEVEL% EQU 0 (
    if "%1"=="-run" (
        mask.exe %2 %3 %4 %5
    )
)