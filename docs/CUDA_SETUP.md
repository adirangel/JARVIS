# CUDA Setup for JARVIS (faster STT)

JARVIS uses **faster-whisper** (CTranslate2) for speech-to-text. GPU acceleration requires **CUDA 12** (`cublas64_12.dll`).

## You have CUDA 13 but need CUDA 12

**faster-whisper needs CUDA 12** – it looks for `cublas64_12.dll`. CUDA 13 uses different DLLs and is not backward compatible.

**Fix:** Install CUDA 12 alongside CUDA 13. Both can coexist.

## Install CUDA 12 on Windows

1. **Download CUDA Toolkit 12.x**
   - https://developer.nvidia.com/cuda-toolkit-archive
   - Choose **CUDA 12.4** or **12.6** (not 13)
   - Select: Windows → x86_64 → Your Windows version → exe (local)

2. **Run the installer**
   - Choose "Express" installation
   - CUDA 12 installs to e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

3. **Ensure CUDA 12 is in PATH (before CUDA 13)**
   - `cublas64_12.dll` lives in: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
   - Add that folder to PATH, or move it **before** `CUDA\v13.x\bin` in PATH
   - **PowerShell (current session):**
     ```powershell
     $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" + $env:PATH
     python verify_voice.py
     ```
   - **Permanent:** System Properties → Environment Variables → edit PATH → add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin` at the top

4. **Restart terminal** and run:
   ```bash
   python verify_voice.py
   ```
   If CUDA works, you won't see "CUDA failed at inference, retrying with CPU".

## If you get cublas64_12.dll error

- **Option A**: Install CUDA 12 (above) – can coexist with CUDA 13
- **Option B**: JARVIS auto-falls back to CPU. No config change needed.

## RTX 4080

RTX 4080 works with CUDA 12. Ensure your NVIDIA driver is up to date.
