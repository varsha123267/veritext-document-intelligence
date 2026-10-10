#!/usr/bin/env bash
# Convenience setup + run script for the Stacks backend.
set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example — add your GEMINI_API_KEY before asking questions."
fi

echo ""
echo "Starting Stacks API on http://localhost:8000"
echo "Open frontend/index.html in your browser once it's running."
echo ""
# No --reload by default: it's more stable for actual use/demos. If you're
# actively editing backend code and want auto-restart on save, run instead:
#   uvicorn app.main:app --reload --reload-dir app --port 8000
uvicorn app.main:app --port 8000
