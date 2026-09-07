"""S3 document-storage provider (Phase 3).

Conforms to the `DocumentStorage` interface from `document_storage.py`.
Authentication uses the standard boto3 credential/provider chain
(local AWS profile/env credentials configured by the operator).
No credentials in source code.

Object-key mapping (simple, configurable prefix; filenames preserved):
    <prefix>/<filename>   e.g. documents/lease.pdf
"""

from document_storage import DocumentStorage


def normalize_prefix(prefix: str) -> str:
    """Return prefix with no leading slash and exactly one trailing slash."""
    p = (prefix or "").strip().lstrip("/")
    if not p:
        return ""
    return p.rstrip("/") + "/"


class S3DocumentStorage(DocumentStorage):
    """S3-backed storage. Reads/writes nothing locally except downloads."""

    def __init__(self, bucket: str, prefix: str = "documents/",
                 region: str = "ap-south-2", client=None):
        if not bucket:
            raise ValueError("S3 bucket name is required (S3_BUCKET).")
        self.bucket = bucket
        self.prefix = normalize_prefix(prefix)
        self.region = region or "ap-south-2"
        self._client = client

    def _s3(self):
        if self._client is not None:
            return self._client
        import boto3

        return boto3.client("s3", region_name=self.region)

    def key_for(self, filename: str) -> str:
        """Map a bare filename to its S3 object key (no subfolders)."""
        name = (filename or "").replace("\\", "/").split("/")[-1]
        if not name:
            raise ValueError("Filename is required.")
        return f"{self.prefix}{name}"

    def _to_key(self, path: str) -> str:
        """Accept a full key or bare filename; return the object key."""
        p = (path or "").replace("\\", "/")
        if self.prefix and p.startswith(self.prefix):
            return p
        return self.key_for(p)

    def _is_pdf_key(self, key: str) -> bool:
        return key.lower().endswith(".pdf")

    def list_documents(self, pattern="**/*.pdf"):
        """List PDF object keys under the prefix (sorted, stable)."""
        keys = []
        client = self._s3()
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self.prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj.get("Key", "")
                if key != self.prefix and self._is_pdf_key(key):
                    keys.append(key)
            token = resp.get("NextContinuationToken")
            if not token:
                break
        return sorted(keys)

    def get_document(self, path: str) -> bytes:
        """Download one object; accept a key or bare filename."""
        resp = self._s3().get_object(Bucket=self.bucket, Key=self._to_key(path))
        body = resp["Body"]
        return body.read() if hasattr(body, "read") else bytes(body)

    def save_document(self, filename: str, data: bytes) -> str:
        """Upload bytes; returns the object key. Never deletes anything."""
        key = self.key_for(filename)
        self._s3().put_object(Bucket=self.bucket, Key=key, Body=data)
        return key

    def delete_document(self, path: str) -> None:
        self._s3().delete_object(Bucket=self.bucket, Key=self._to_key(path))

    def describe(self) -> str:
        return f"S3 s3://{self.bucket}/{self.prefix}"


def get_s3_store(config=None, client=None):
    """Build an S3DocumentStorage from config (bucket required)."""
    bucket = ""
    prefix = "documents/"
    region = "ap-south-2"
    if config is not None:
        bucket = getattr(config, "s3_bucket", bucket)
        prefix = getattr(config, "s3_prefix", prefix)
        region = getattr(config, "s3_region", region)
    return S3DocumentStorage(bucket=bucket, prefix=prefix, region=region,
                             client=client)
