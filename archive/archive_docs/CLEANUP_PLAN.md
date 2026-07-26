#!/bin/bash
# SRF Project Cleanup Script
# Removes unwanted logs, temporary files, and organizes remaining files

echo "============================================================"
echo "SRF PROJECT CLEANUP - Safe File Removal"
echo "============================================================"

# Safe cleanup - remove only files that are definitely not needed
echo "🧹 Cleaning up temporary and log files..."

# Remove large log files from root
find . -maxdepth 1 -name "*.log" -type f -size +1M -exec rm -v {} \;

# Remove temporary monitoring logs
find . -maxdepth 1 -name "*monitor*.log" -type f -exec rm -v {} \;

# Remove Python cache files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Remove editor swap files
find . -name "*.swp" -delete
find . -name "*.swo" -delete

echo "✅ Basic cleanup completed"
echo ""
echo "📊 Files removed:"
echo "  - Large log files (>1MB)"
echo "  - Monitoring logs"
echo "  - Python cache files"
echo "  - Editor swap files"
echo ""
echo "⚠️  Remaining files organized by category:"
ls -1 *.log *.sh 2>/dev/null | head -20