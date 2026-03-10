@echo off
chcp 65001 >nul
set "VENV_DIR=venv"

echo ===================================================
echo 슬롯 게임 데이터 추출기 로딩 (Slot Game Extractor)
echo ===================================================

:: Python 설치 여부 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python이 시스템에 설치되어 있지 않거나 환경변수 PATH가 등록되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python 최신 버전을 다운로드하고
    echo "Add Python to PATH" 체크박스를 선택하여 설치해주세요.
    pause
    exit /b 1
)

:: 가상 환경이 없으면 생성
if not exist "%VENV_DIR%" (
    echo.
    echo 첫 실행 초기화 중입니다. 애플리케이션 가상 환경을 생성합니다...
    python -m venv "%VENV_DIR%"
)

:: 가상 환경 활성화
echo.
echo 가상 환경 활성화 중...
call "%VENV_DIR%\Scripts\activate.bat"

:: requirements.txt 확인 후 의존성 설치/업데이트 상태 확인
echo.
echo 패키지 의존성을 설치합니다. (처음 실행 시 시간이 소요될 수 있습니다)
python -m pip install --upgrade pip >nul

echo.
echo [중요] RTX 그래픽카드(GPU) 가속 지원 모듈을 설치합니다...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 나머지 패키지들을 설치 중입니다...
pip install -r requirements.txt

:: 메인 프로그램 실행
echo.
echo 애플리케이션을 시작합니다...
echo.
python main_app.py

exit
