# AI-PIPELINE setup script for Windows
# Run once after git clone: .\setup.ps1

Write-Host "Installing Python dependencies..."
pip install -r requirements.txt

Write-Host "Initializing CodeGraph (requires Node.js)..."
npx @colbymchenry/codegraph init
npx @colbymchenry/codegraph index

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host ""
    Write-Host "Created .env from .env.example"
    Write-Host "IMPORTANT: Edit .env and fill in your GCP credentials before running."
} else {
    Write-Host ".env already exists — skipping."
}

Write-Host ""
Write-Host "Setup complete! Run: streamlit run app.py"
