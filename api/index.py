"""Vercel entry point: exposes the Flask WSGI app. All /api/* requests are rewritten here (see vercel.json)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402,F401
