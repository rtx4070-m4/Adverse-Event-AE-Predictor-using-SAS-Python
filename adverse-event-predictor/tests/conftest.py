"""
Pytest configuration and shared fixtures.
File: tests/conftest.py
"""
import sys
from pathlib import Path

# Add project directories to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "api"))
