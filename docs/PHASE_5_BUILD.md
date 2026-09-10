# Phase 5 Build Policy (design — no deployment change in 5.0)

## Findings (audited 2026-09-10)

1. EC2 runs **python3.11** (`setup_ec2.sh` installs `python3.11`);
   local dev here is 3.12. The 3.11 gate (`tests/test_py311_compat.py`)
   exists precisely because of this gap.
2. `setup_ec2.sh` installs from **`requirements.txt` (unpinned)**,
   not the lock file — every deploy can drift.
3. `requirements.lock.txt` is **UTF-16 encoded** (PowerShell redirect
   artifact; pip expects UTF-8) and **stale/partial**: 152 pins but
   missing `psycopg2-binary`, `pgvector`, and `boto3`, all required by
   `requirements.txt`. It also pins `torch==2.9.1` + full ML stack.
4. `langchain-aws` (needed for 5A Bedrock) is in NEITHER file.

## Intended approach (5A implements)

1. Regenerate the lock on the **EC2 Python version** (3.11):
   `pip freeze` after a clean `requirements.txt` install + the 5A
   additions (`langchain-aws`, `boto3` already needed), written as
   **UTF-8**, committed.
2. `setup_ec2.sh` installs with `pip install -r requirements.lock.txt`
   (keep a `--upgrade pip` first; keep idempotency).
3. Drift detection before deployment: CI (and a pre-deploy local step)
   runs `pip freeze` in a clean 3.11 venv and diffs package==version
   lines against the lock; any drift fails the build BEFORE scp.
4. The Python 3.11 compatibility gate remains and runs in the same
   pre-deploy step (it already runs in the unit suite).
5. Python version match is enforced by step 3's environment (3.11
   venv); `setup_ec2.sh` keeps installing `python3.11` explicitly.

No script changes in 5.0 — this document is the spec 5A executes.
