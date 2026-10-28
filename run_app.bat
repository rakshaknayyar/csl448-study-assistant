@echo off
rem Starts the CSL448 Study Assistant. Double-click this file, or run: run_app.bat
rem - uses the project's virtual environment (venv), not the system Python
rem - reads GEMINI_API_KEY from your Windows user settings (saved with setx),
rem   so it works even in a window opened before the key was saved
rem - builds the search index and the React app first if they are missing
rem Stop the app with Ctrl+C in this window.

cd /d "%~dp0"

for /f "tokens=2,*" %%a in ('reg query HKCU\Environment /v GEMINI_API_KEY 2^>nul ^| find "GEMINI_API_KEY"') do set "GEMINI_API_KEY=%%b"
if not defined GEMINI_API_KEY echo WARNING: GEMINI_API_KEY is not set. Answers will fail until you run: setx GEMINI_API_KEY "your-key"

if not exist index\faiss.index (
    echo Building the index from notes\ first ...
    venv\Scripts\python.exe ingest.py
)

if not exist web\dist\index.html (
    echo Building the React app first ^(needs Node.js^) ...
    pushd web
    call npm install
    call npm run build
    popd
)

echo.
echo  CSL448 Study Assistant is starting. It opens in your browser at http://localhost:8501
echo  Keep this window open while you use the app. Press Ctrl+C here to stop it.
echo.
venv\Scripts\python.exe server.py
