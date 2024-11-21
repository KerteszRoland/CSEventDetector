@echo off
nvcc mask.cu -o mask.exe -lgdi32 -luser32 -O2
if %ERRORLEVEL% EQU 0 (
    if "%1"=="-run" (
        mask.exe
    )
)