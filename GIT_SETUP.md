# Git Setup Guide

## Quick Start - Push to GitHub

### Step 1: Initialize Git Repository
```bash
cd "C:\Users\tejad\OneDrive\Desktop\ML Project\Polynomial_Regression_Project"
git init
```

### Step 2: Add All Files
```bash
git add .
```

### Step 3: Create Initial Commit
```bash
git commit -m "Initial commit: Ice Cream Sales Predictor with Polynomial Regression"
```

### Step 4: Create GitHub Repository
1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Name it: `ice-cream-sales-predictor` (or any name you prefer)
4. **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### Step 5: Connect to GitHub
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/ice-cream-sales-predictor.git
```

### Step 6: Push to GitHub
```bash
git branch -M main
git push -u origin main
```

## Alternative: Using SSH
If you prefer SSH:
```bash
git remote add origin git@github.com:YOUR_USERNAME/ice-cream-sales-predictor.git
git push -u origin main
```

## Complete Command Sequence
```bash
# Navigate to project
cd "C:\Users\tejad\OneDrive\Desktop\ML Project\Polynomial_Regression_Project"

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Ice Cream Sales Predictor - Polynomial Regression Model"

# Add remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Files Included
- ✅ app.py (Main application)
- ✅ run_app.py (Launcher)
- ✅ run.bat (Windows launcher)
- ✅ Temperature_vs_IceCreamSales.csv (Dataset)
- ✅ requirements.txt (Dependencies)
- ✅ README.md (Documentation)
- ✅ .gitignore (Git ignore file)

## Files Excluded (via .gitignore)
- ❌ __pycache__/ (Python cache)
- ❌ *.pyc (Compiled Python files)
- ❌ .venv/ (Virtual environment)
- ❌ .vscode/ (IDE settings)

## Troubleshooting

**Issue:** "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

**Issue:** Authentication required
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys

**Issue:** "failed to push some refs"
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

