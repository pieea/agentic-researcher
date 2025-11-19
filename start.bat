@echo off
REM Market Research Platform - Windows 실행 스크립트
REM 백엔드와 프론트엔드를 동시에 실행합니다

echo 🚀 Market Research Platform 시작...
echo.

REM 환경 변수 체크
if not exist "backend\.env" (
    echo ⚠️  backend\.env 파일이 없습니다!
    echo backend\.env.example을 복사하여 backend\.env를 만들고 API 키를 설정하세요.
    echo.
    echo 예시:
    echo   copy backend\.env.example backend\.env
    echo   # backend\.env 파일을 편집하여 API 키 입력
    exit /b 1
)

REM 백엔드 의존성 체크
echo 📦 백엔드 의존성 확인 중...
cd backend
if not exist ".venv" (
    echo 가상환경 생성 및 의존성 설치 중...
    uv venv
    uv pip install -e ".[dev]"
) else (
    echo ✅ 백엔드 의존성 준비 완료
)
cd ..

REM 프론트엔드 의존성 체크
echo 📦 프론트엔드 의존성 확인 중...
cd frontend
if not exist "node_modules" (
    echo 프론트엔드 의존성 설치 중...
    call npm install
) else (
    echo ✅ 프론트엔드 의존성 준비 완료
)
cd ..

echo.
echo ✅ 모든 의존성 준비 완료
echo.

REM 로그 디렉토리 생성
if not exist "logs" mkdir logs

echo 🔧 백엔드 서버 시작 중... (포트 8000)
cd backend
start /B cmd /c "uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload > ..\logs\backend.log 2>&1"
cd ..

echo ⏳ 백엔드 서버 준비 대기 중...
timeout /t 3 /nobreak >nul

echo 🎨 프론트엔드 서버 시작 중... (포트 3000)
cd frontend
start /B cmd /c "npm run dev > ..\logs\frontend.log 2>&1"
cd ..

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ✨ 서버가 성공적으로 시작되었습니다!
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 📱 프론트엔드: http://localhost:3000
echo 🔧 백엔드 API: http://localhost:8000
echo 📖 API 문서: http://localhost:8000/docs
echo.
echo 💡 팁:
echo   - 백엔드 로그: type logs\backend.log
echo   - 프론트엔드 로그: type logs\frontend.log
echo.
echo 종료하려면 이 창을 닫으세요
echo.

pause
