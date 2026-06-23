"""Console formatting utilities for logging.

This is a stub that provides basic formatting functions.
The original module likely uses `rich` for colored console output.
"""

def warn(msg: str) -> str:
    """Format a warning message."""
    return f"[WARN] {msg}"

def info(msg: str) -> str:
    """Format an info message."""
    return f"[INFO] {msg}"

def error(msg: str) -> str:
    """Format an error message."""
    return f"[ERROR] {msg}"

def ok(msg: str) -> str:
    """Format a success/ok message."""
    return f"[OK] {msg}"
