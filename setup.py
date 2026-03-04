"""JARVIS Setup — installs all dependencies in one command.

Usage:
    python setup.py
"""

import subprocess
import sys
import os


def main():
    print("=" * 50)
    print("  J.A.R.V.I.S — Setup")
    print("=" * 50)
    print()

    # Step 1: pip install
    print("[1/2] Installing Python dependencies...")
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n❌ pip install failed. Check errors above.")
        sys.exit(1)
    print("✅ Python packages installed.\n")

    # Step 2: Playwright browsers (for browser automation tools)
    print("[2/2] Installing Playwright browser binaries...")
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("⚠️  Playwright install failed — browser tools won't work.")
        print("   You can install manually later: python -m playwright install\n")
    else:
        print("✅ Playwright browsers installed.\n")

    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("config", exist_ok=True)

    print("=" * 50)
    print("  ✅ Setup complete!")
    print()
    print("  Run JARVIS:  python main.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
