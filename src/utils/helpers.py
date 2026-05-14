import hashlib
from typing import List

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of size chunk_size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
