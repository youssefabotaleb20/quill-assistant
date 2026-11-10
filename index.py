"""Vercel entrypoint: exposes the FastAPI app from backend/app/main.py.

Vercel looks for an `app` object in a root file such as index.py. The code
lives in backend/ (imported as the `app` package), so add it to the path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.main import app

__all__ = ["app"]
