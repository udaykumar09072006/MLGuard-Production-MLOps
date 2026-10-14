"""Safe identifier and hashing utilities."""

import hashlib
import uuid


def generate_unique_id(prefix: str = "") -> str:
    """Generate safe, URL-friendly random unique identifier."""
    token = uuid.uuid4().hex[:12]
    return f"{prefix}_{token}" if prefix else token


def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA256 digest of a file to fingerprint data versions."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
