"""Unit tests for the pure parts (no GPU, model weights or data needed).

    python test.py
"""
import sys
import unittest

if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover("tests"))
    sys.exit(0 if result.wasSuccessful() else 1)
