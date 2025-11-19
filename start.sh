#!/bin/bash

# Market Research Platform - 통합 실행 스크립트
# 백엔드와 프론트엔드를 동시에 실행합니다

set -e  # 에러 발생시 중단

echo "🚀 Market Research Platform 시작..."
echo ""

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 환경 변수 체크
if [ ! -f "backend/.env" ]; then
    echo -e "${RED}⚠️  backend/.env 파일이 없습니다!${NC}"
    echo "backend/.env.example을 복사하여 backend/.env를 만들고 API 키를 설정하세요."
    echo ""
    echo "예시:"
    echo "  cp backend/.env.example backend/.env"
    echo "  # backend/.env 파일을 편집하여 API 키 입력"
    exit 1
fi

# 백엔드 의존성 체크 및 설치
echo -e "${BLUE}📦 백엔드 의존성 확인 중...${NC}"
cd backend
if [ ! -d ".venv" ]; then
    echo "가상환경 생성 및 의존성 설치 중..."
    uv venv
    uv pip install -e ".[dev]"
else
    echo "✅ 백엔드 의존성 준비 완료"
fi
cd ..

# 프론트엔드 의존성 체크 및 설치
echo -e "${BLUE}📦 프론트엔드 의존성 확인 중...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo "프론트엔드 의존성 설치 중..."
    npm install
else
    echo "✅ 프론트엔드 의존성 준비 완료"
fi
cd ..

echo ""
echo -e "${GREEN}✅ 모든 의존성 준비 완료${NC}"
echo ""

# 로그 디렉토리 생성
mkdir -p logs

# 트랩 설정 (Ctrl+C 시 모든 프로세스 종료)
trap 'echo -e "\n${RED}🛑 서버 종료 중...${NC}"; kill 0' EXIT INT TERM

# 백엔드 시작
echo -e "${BLUE}🔧 백엔드 서버 시작 중... (포트 8000)${NC}"
cd backend
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 백엔드가 준비될 때까지 대기
echo "⏳ 백엔드 서버 준비 대기 중..."
sleep 3

# 백엔드 헬스체크
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 백엔드 서버 준비 완료${NC}"
else
    echo -e "${RED}⚠️  백엔드 서버가 정상적으로 시작되지 않았습니다${NC}"
    echo "logs/backend.log를 확인하세요"
fi

# 프론트엔드 시작
echo -e "${BLUE}🎨 프론트엔드 서버 시작 중... (포트 3000)${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ 서버가 성공적으로 시작되었습니다!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "📱 프론트엔드: ${BLUE}http://localhost:3000${NC}"
echo -e "🔧 백엔드 API: ${BLUE}http://localhost:8000${NC}"
echo -e "📖 API 문서: ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "💡 팁:"
echo "  - 백엔드 로그: tail -f logs/backend.log"
echo "  - 프론트엔드 로그: tail -f logs/frontend.log"
echo ""
echo -e "${RED}종료하려면 Ctrl+C를 누르세요${NC}"
echo ""

# 프로세스 대기
wait
