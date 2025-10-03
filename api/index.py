import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the Flask app
from app import app as application

# Vercel serverless function handler
def handler(request, context):
    return application(request.environ, context)
