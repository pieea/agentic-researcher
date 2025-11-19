#!/bin/bash

# Market Research Platform - 서버 종료 스크립트

echo "🛑 서버 종료 중..."

# 백엔드 종료 (포트 8000)
BACKEND_PID=$(lsof -ti:8000)
if [ ! -z "$BACKEND_PID" ]; then
    echo "백엔드 서버 종료 중... (PID: $BACKEND_PID)"
    kill -9 $BACKEND_PID
    echo "✅ 백엔드 서버 종료 완료"
else
    echo "⚠️  실행 중인 백엔드 서버가 없습니다"
fi

# 프론트엔드 종료 (포트 3000, 5173)
FRONTEND_PID=$(lsof -ti:3000,5173)
if [ ! -z "$FRONTEND_PID" ]; then
    echo "프론트엔드 서버 종료 중... (PID: $FRONTEND_PID)"
    kill -9 $FRONTEND_PID
    echo "✅ 프론트엔드 서버 종료 완료"
else
    echo "⚠️  실행 중인 프론트엔드 서버가 없습니다"
fi

echo ""
echo "✅ 모든 서버가 종료되었습니다"
