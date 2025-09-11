# utils.py
import os

def safe_makedirs(path: str, force: bool = False) -> None:
    """
    Create directory safely.
    If file with same name exists, overwrite only if force=True.
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            return
        if force:
            os.remove(path)
            os.makedirs(path, exist_ok=True)
            return
        raise FileExistsError(
            f"Path already exists and is not a directory: {path!r}. "
            f"Rename it, remove it, or call safe_makedirs(path, force=True)."
        )
    os.makedirs(path, exist_ok=True)
