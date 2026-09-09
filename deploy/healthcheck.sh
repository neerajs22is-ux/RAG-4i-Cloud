#!/bin/bash
# Readiness probe for the EC2 pilot (systemd service: rag4i-cloud).
# Checks: (1) Streamlit HTTP health, (2) knowledge-base readiness,
# (3) LLM reachability — via the app's own status helpers, no mocks.
# Exit 0 only when the app can actually answer questions.
# Usage: APP_DIR=/home/ec2-user/RAG-4i-Cloud ./deploy/healthcheck.sh
set -u
APP_DIR="${APP_DIR:-/home/ec2-user/RAG-4i-Cloud}"

curl -sf -m 10 http://localhost:8501/_stcore/health > /dev/null || {
  echo "UNHEALTHY: streamlit not responding on :8501"; exit 1; }

export APP_DIR_CHECK="$APP_DIR"
"$APP_DIR/venv/bin/python" - <<'EOF'
import os
import sys

APP_DIR = os.environ["APP_DIR_CHECK"]
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)
from backend import check_llm_status, get_knowledge_base_status

kb = get_knowledge_base_status()
llm = check_llm_status()
print("KB_READY:", kb.get("ready"), "LLM:", llm.get("reachable"))
sys.exit(0 if (kb.get("ready") and llm.get("reachable")) else 2)
EOF
