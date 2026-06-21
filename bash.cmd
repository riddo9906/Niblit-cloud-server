@echo off
setlocal
set "BASH_EXE="
if exist "%ProgramFiles%\Git\bin\bash.exe" (
  set "BASH_EXE=%ProgramFiles%\Git\bin\bash.exe"
) else if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" (
  set "BASH_EXE=%ProgramFiles(x86)%\Git\bin\bash.exe"
) else (
  where bash.exe >nul 2>&1
  if not errorlevel 1 (
    for /f "usebackq delims=" %%I in (`where bash.exe`) do (
      set "BASH_EXE=%%I"
      goto :run
    )
  )
)

:run
if not defined BASH_EXE (
  echo bash not found >&2
  exit /b 1
)
"%BASH_EXE%" %*
