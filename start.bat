@echo on
set "VENV_DIR=venv"

echo ===================================================
echo DEBUG MODE: Slot Game Data Extractor
echo ===================================================

:: 1. Python check
python --version
if %errorlevel% neq 0 (
    echo [CRITICAL] Python not found!
    pause
    exit /b 1
)

:: 2. VENV creation
if not exist "%VENV_DIR%" (
    echo [DEBUG] Creating VENV...
    python -m venv "%VENV_DIR%"
)

:: 3. Activation
echo [DEBUG] Activating VENV...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [CRITICAL] Activation failed!
    pause
    exit /b 1
)

:: 4. PIP Upgrade
echo [DEBUG] Upgrading PIP...
python -m pip install --upgrade pip

:: 5. Torch Check
echo [DEBUG] Checking Torch...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: 6. ONNX Check
echo [DEBUG] Installing ONNX...
python -m pip install onnxruntime-gpu

:: 7. Requirements
echo [DEBUG] Installing Requirements...
python -m pip install -r requirements.txt

:: 8. Launch
echo [DEBUG] Launching App...
python main_app.py

if %errorlevel% neq 0 (
    echo [CRITICAL] App crashed with error code %errorlevel%
    pause
)

pause
