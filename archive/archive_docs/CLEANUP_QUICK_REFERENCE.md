# 🧹 Quick Cleanup Guide - SRF Project

## 🎯 **What This Cleanup Does**

The cleanup script will:
1. **Remove unwanted files** (logs, cache, temporary files)
2. **Organize scripts** into logical folders (28+ scripts!)
3. **Organize documentation** into clear categories
4. **Archive old results** to keep your directory clean

## 🚀 **How to Use**

### **Option 1: Full Automatic Cleanup** (Recommended)
```bash
chmod +x cleanup_and_organize.sh
./cleanup_and_organize.sh
```

### **Option 2: Manual Cleanup** (If you prefer control)
```bash
# Just remove unwanted logs
rm *.log

# Then organize scripts manually
mkdir scripts_launch scripts_run scripts_check
mv launch_*.sh scripts_launch/
mv run_*.sh scripts_run/
mv check_*.sh scripts_check/
```

## 📋 **What Gets Deleted** (Safe to Remove)

### **❌ DELETE These:**
- **All `.log` files** - Temporary execution logs
- **`*.pyc` files** - Python bytecode cache
- **`__pycache__/`** - Python cache directories
- **`*.swp`, `*.swo`** - Editor swap files

### **✅ KEEP These:**
- **All `.md` documentation files** - Important for reference
- **All `.py` scripts** - Core SRF implementation
- **`scripts/` folder** - Core utilities
- **`info/` folder** - Documentation
- **`srf/` folder** - Core implementation
- **Current `results/`** - Your latest results

## 📁 **New Folder Structure After Cleanup**

```
lmms-eval/
├── scripts_launch/      # 7 files - Launch experiments
├── scripts_run/         # 12 files - Run evaluations  
├── scripts_check/       # 4 files - Check status
├── scripts_monitor/     # 2 files - Monitor experiments
├── scripts_test/        # 2 files - Testing utilities
├── docs_status/         # Status & progress tracking
├── docs_analysis/       # Analysis & investigation docs
├── docs_guides/         # Usage guides & references
├── results_archive/     # Old experiment results
├── results/             # Current results (keep working)
├── srf/                 # Core SRF implementation (keep!)
├── info/                # Documentation (keep!)
└── my_analysis/         # Analysis tools (keep!)
```

## ⚠️ **Before You Clean Up**

### **1. Check for Important Unsaved Work**
```bash
# Check git status
git status

# Check for any important unstaged work
git diff --name-only
```

### **2. Backup Important Results** (Optional)
```bash
# If you want to backup current results first
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/
```

### **3. Make Sure You Don't Need Current Logs**
```bash
# Check if any important info in logs
grep -i "error\|success\|best\|final" *.log | head -10
```

## 🎯 **Quick Clean vs Deep Clean**

### **Quick Clean** (Safe, Recommended)
```bash
# Just remove logs and cache
rm *.log
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### **Deep Clean** (Use the script)
```bash
# Full cleanup + organization
./cleanup_and_organize.sh
```

## 🔄 **After Cleanup**

### **Check What's Left**
```bash
# See root directory is now clean
ls -la | grep -v "^d"
ls -la | grep "^d" | head -15
```

### **Test That Everything Still Works**
```bash
# Test that SRF still works
python -c "from srf.srf import SRF; print('✅ SRF works')"

# Check scripts are accessible
bash scripts_check/check_sweep_detailed.sh
```

## 📊 **Space Savings**

Typical cleanup results:
- **Logs removed**: ~50-200MB
- **Cache files**: ~10-50MB  
- **Archive old results**: ~100-500MB
- **Total space saved**: ~200-800MB

## 🚨 **What NOT to Delete**

### **❌ DON'T Delete:**
- **`srf/`** folder - Core implementation
- **`info/`** folder - Documentation
- **`my_analysis/qwen_attn_patch.py`** - Main patching file
- **`scripts/`** folder - Core utilities
- **Current `.md` files** - Documentation
- **`results/srf_focused_sweep/`** - Latest experimental results

## 🎉 **After Cleanup**

Your directory will be:
- ✅ **Organized**: Everything in logical folders
- ✅ **Clean**: No unwanted logs or temporary files
- ✅ **Navigable**: Easy to find what you need
- ✅ **Functional**: All important files preserved

## 🚀 **Ready to Clean Up?**

```bash
chmod +x cleanup_and_organize.sh
./cleanup_and_organize.sh
```

**This will make your project much easier to work with!** 🎯