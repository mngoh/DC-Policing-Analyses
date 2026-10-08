"""Unit tests run without a database: clear DATABASE_URL before any project import."""
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
