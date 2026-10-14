"""Controlled file storage utilities with path traversal defense."""

import os

from app.core.exceptions import DatasetValidationError


def sanitize_filename(filename: str) -> str:
    """Strip dangerous characters and directory traversals."""
    basename = os.path.basename(filename)
    clean = "".join(c for c in basename if c.isalnum() or c in (".", "_", "-"))
    return clean or "unnamed_dataset.csv"


def secure_save_path(base_dir: str, filename: str) -> str:
    """Ensure path is within intended base directory and prevents directory traversal."""
    sanitized = sanitize_filename(filename)
    full_path = os.path.abspath(os.path.join(base_dir, sanitized))
    base_abs = os.path.abspath(base_dir)

    if not full_path.startswith(base_abs):
        raise DatasetValidationError("Illegal path traversal attempt detected.")

    return full_path
