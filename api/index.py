import sys
import os

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app

# This is the entry point for Vercel
# Vercel will use this app object
app = app
