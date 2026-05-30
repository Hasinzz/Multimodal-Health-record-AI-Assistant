param(
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    throw "This script must be run from the repository root."
}

$currentBranch = git rev-parse --abbrev-ref HEAD
if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine the current branch."
}

if ($currentBranch -eq "HEAD") {
    throw "Detached HEAD state detected. Check out a branch before syncing."
}

Write-Host "Fetching latest changes from $Remote..."
git fetch $Remote
if ($LASTEXITCODE -ne 0) {
    throw "Git fetch failed."
}

if (-not $Branch) {
    $Branch = $currentBranch
}

if ($Branch -ne $currentBranch) {
    Write-Host "Note: syncing branch $currentBranch using remote branch $Branch."
}

Write-Host "Rebasing local changes onto $Remote/$Branch..."
git pull --rebase --autostash $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    throw "Git pull --rebase failed. Resolve conflicts, then run the script again."
}

Write-Host "Sync complete."