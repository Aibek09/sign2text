#!/usr/bin/env python3
"""Launch the app with its installed model environment, regardless of shell setup."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
python = root / "work" / "model-env" / "bin" / "python"
if not python.is_file():
    sys.exit("Model environment is missing. Follow installation instructions in README.md first.")
os.execv(str(python), [str(python), str(root / "app" / "server.py"), *sys.argv[1:]])
