#!/bin/bash
# EC2 setup for RAG-4i-Cloud pilot (Amazon Linux 2023, existing instance).
# Run once as ec2-user. Idempotent. No secrets in this script.
set -euo pipefail

APP_DIR="$HOME/RAG-4i-Cloud"

echo "--- system packages ---"
sudo dnf install -y python3.11 python3.11-pip git

echo "--- repository ---"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone https://github.com/neerajs22is-ux/RAG-4i-Cloud.git "$APP_DIR"
fi

echo "--- virtualenv + dependencies ---"
python3.11 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "--- streamlit config ---"
mkdir -p "$HOME/.streamlit"
cp "$APP_DIR/deploy/streamlit_config.toml" "$HOME/.streamlit/config.toml"

echo "--- systemd service ---"
sudo cp "$APP_DIR/deploy/rag4i-cloud.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rag4i-cloud.service

echo "--- next (manual, on the instance) ---"
echo "1. Copy deploy/env.ec2.example to $APP_DIR/.env and set DB_PASSWORD."
echo "2. sudo systemctl start rag4i-cloud.service"
echo "3. journalctl -u rag4i-cloud.service -f   (logs)"
