#!/usr/bin/env bash
# AI-PIPELINE setup script for Linux/Mac
# Run once after git clone: bash setup.sh

set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Initializing CodeGraph (requires Node.js)..."
npx @colbymchenry/codegraph init
npx @colbymchenry/codegraph index

if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from .env.example"
    echo "IMPORTANT: Edit .env and fill in your GCP credentials before running."
else
    echo ".env already exists — skipping."
fi

echo ""
echo "Setup complete! Run: streamlit run app.py"
