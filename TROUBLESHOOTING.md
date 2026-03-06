# Troubleshooting Guide

## Windows DLL Error: "A dynamic link library (DLL) initialization routine failed"

If you encounter this error when running `streamlit run app.py` or importing torch:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. 
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

### Cause

This is caused by missing Visual C++ Redistributables on Windows. PyTorch requires the Microsoft Visual C++ Redistributable for Visual Studio 2015-2022.

### Solution 1: Install Visual C++ Redistributables (Recommended)

1. Download the latest Visual C++ Redistributable from Microsoft:
   - **x64**: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - **ARM64**: https://aka.ms/vs/17/release/vc_redist.arm64.exe

2. Run the installer and restart your computer

3. Try running the app again:
   ```bash
   streamlit run app.py
   ```

### Solution 2: Use Conda/Miniconda (Alternative)

If the above doesn't work, use Conda which handles dependencies better:

```bash
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# Create a new environment
conda create -n sourcesleuth python=3.11
conda activate sourcesleuth

# Install PyTorch CPU version
conda install pytorch cpuonly -c pytorch

# Install SourceSleuth
pip install -e ".[dev,ui]"
```

### Solution 3: Use ONNX Runtime (Lightweight Alternative)

For CPU-only inference, you can use ONNX runtime instead of PyTorch:

```bash
pip install optimum[onnxruntime]
```

Then modify `src/vector_store.py` to use ONNX backend.

### Solution 4: Check CPU Compatibility

Older CPUs (pre-2011) may not support AVX instructions required by PyTorch. Check your CPU:

```bash
# Windows PowerShell
python -c "import platform; print(platform.processor())"
```

If your CPU doesn't support AVX, you'll need to:
- Use a cloud-based solution (Google Colab, Kaggle)
- Build PyTorch from source for your specific CPU
- Use an alternative embedding library

### Solution 5: Python Version Compatibility

Python 3.13+ may have compatibility issues with some ML libraries. Use Python 3.10-3.12:

```bash
# Create a new virtual environment with Python 3.11
python -m venv .venv --python=python3.11
.venv\Scripts\activate
pip install -e ".[dev,ui]"
```

## Verification

After applying any solution, verify the installation:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully')"
python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers OK')"
```

## Still Having Issues?

1. Check the [GitHub Issues](https://github.com/Ishwarpatra/OpenSourceSleuth/issues) for similar problems
2. Run with verbose logging:
   ```bash
   set SOURCESLEUTH_LOG_LEVEL=DEBUG
   streamlit run app.py
   ```
3. Provide the following information when filing an issue:
   - Windows version (`winver`)
   - Python version (`python --version`)
   - CPU model (from Task Manager > Performance > CPU)
   - Full error traceback
