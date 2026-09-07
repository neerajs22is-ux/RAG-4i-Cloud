"""Copy-only migration of the small test set to S3 (lease/contract only).

- Locates local lease.pdf + contract.pdf under the given folder.
- Uploads missing keys to <prefix>/<filename>; never overwrites existing
  keys, never deletes local originals, never deletes S3 objects.
- Reports found/uploaded/skipped/failed + exact keys created.

Usage:
    python migrate_to_s3.py <local-folder>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config  # noqa: E402

TEST_FILES = ["lease.pdf", "contract.pdf"]


def migrate(folder):
    from document_storage import LocalDocumentStorage
    from s3_document_storage import get_s3_store

    cfg = get_config()
    if (getattr(cfg, "document_storage", "local") or "local").lower() != "s3" \
            and not getattr(cfg, "s3_bucket", ""):
        print("Note: DOCUMENT_STORAGE is not 's3'; using S3_BUCKET anyway.")
    if not getattr(cfg, "s3_bucket", ""):
        return False, "S3 bucket is not configured (S3_BUCKET)."
    if not folder or not os.path.isdir(folder):
        return False, "Folder path is not a directory."

    local = LocalDocumentStorage(folder)
    s3 = get_s3_store(cfg)
    try:
        existing = set(s3.list_documents())
    except Exception as e:
        return False, f"Cannot list S3 bucket (check credentials): {e}"

    report = {"found": 0, "uploaded": 0, "skipped": 0, "failed": 0,
              "keys": [], "failed_files": []}
    for name in TEST_FILES:
        src = os.path.join(folder, name)
        if not os.path.isfile(src):
            continue
        report["found"] += 1
        key = s3.key_for(name)
        if key in existing:
            report["skipped"] += 1
            continue
        try:
            with open(src, "rb") as f:
                created = s3.save_document(name, f.read())
            report["uploaded"] += 1
            report["keys"].append(created)
        except Exception as e:
            report["failed"] += 1
            report["failed_files"].append(name)
            print(f"Error uploading {name}: {e}")
    msg = (f"Found={report['found']} uploaded={report['uploaded']} "
           f"skipped={report['skipped']} failed={report['failed']}. "
           f"Keys: {report['keys'] or 'none'}.")
    if report["failed_files"]:
        msg += f" Failed: {report['failed_files']}."
    return report["failed"] == 0, msg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder")
    args = ap.parse_args()
    ok, msg = migrate(args.folder)
    print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
