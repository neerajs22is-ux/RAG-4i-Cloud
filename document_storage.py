"""Document storage provider abstraction (local default + S3 opt-in).

Use get_document_storage() to select by configuration:
DOCUMENT_STORAGE=local (default) or s3. No code changes to switch.
"""

import glob
import os
import shutil
from abc import ABC, abstractmethod
from typing import List


class DocumentStorage(ABC):
    """Minimal document-storage interface for future S3 migration."""

    @abstractmethod
    def list_documents(self, pattern="**/*.pdf"):
        """Return absolute paths of documents matching pattern."""
        raise NotImplementedError

    @abstractmethod
    def get_document(self, path):
        """Return raw bytes for a stored document."""
        raise NotImplementedError

    @abstractmethod
    def save_document(self, filename, data: bytes):
        """Persist bytes under filename, return absolute path."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, path):
        """Delete the document at path."""
        raise NotImplementedError


class LocalDocumentStorage(DocumentStorage):
    """Filesystem-backed storage. `base_dir` is the folder to scan."""

    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)

    def list_documents(self, pattern="**/*.pdf") -> List[str]:
        if not os.path.isdir(self.base_dir):
            return []
        glob_path = os.path.join(self.base_dir, pattern)
        files = glob.glob(glob_path, recursive=True)
        # Return absolute, file-only paths in stable sorted order.
        return sorted(
            [os.path.abspath(p) for p in files if os.path.isfile(p)]
        )

    def _contained(self, path: str) -> str:
        """Absolute path, verified to stay inside base_dir (no traversal)."""
        abs_path = os.path.abspath(path)
        if os.path.commonpath([self.base_dir, abs_path]) != self.base_dir:
            raise ValueError(f"Path escapes the document folder: {path}")
        return abs_path

    def get_document(self, path: str) -> bytes:
        with open(self._contained(path), "rb") as f:
            return f.read()

    def save_document(self, filename: str, data: bytes) -> str:
        dest = self._contained(os.path.join(self.base_dir, filename))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return dest

    def delete_document(self, path: str) -> None:
        abs_path = self._contained(path)
        if os.path.isfile(abs_path):
            os.remove(abs_path)

    def ensure_exists(self) -> bool:
        return os.path.isdir(self.base_dir)


def validate_local_path(path: str, must_exist: bool = True) -> str:
    """Validate a local folder path; returns its absolute form.

    Raises ValueError on empty paths, NUL bytes, or (when must_exist)
    missing/non-directory paths. No silent acceptance.
    """
    if not path or not path.strip() or "\x00" in path:
        raise ValueError("Folder path is required.")
    abs_path = os.path.abspath(path)
    if must_exist and not os.path.isdir(abs_path):
        raise ValueError(f"Folder path is not a directory: {path}")
    return abs_path


def validate_s3_prefix(prefix: str) -> str:
    """Validate an S3 prefix (no leading slash, no empty keys, no '..')."""
    from s3_document_storage import normalize_prefix

    if prefix is None:
        raise ValueError("S3 prefix is required.")
    if ".." in prefix.replace("\\", "/").split("/"):
        raise ValueError("S3 prefix must not contain '..'.")
    return normalize_prefix(prefix)


def get_document_storage(config=None, folder=None):
    """Factory dispatched by config: DOCUMENT_STORAGE=local (default) or s3."""
    name = "local"
    if config is not None:
        name = (getattr(config, "document_storage", name) or name).lower()
    if name == "s3":
        from s3_document_storage import get_s3_store

        return get_s3_store(config)
    return LocalDocumentStorage(folder or ".")
