#!/bin/bash
cd "$(dirname "$0")"
source ../.venv/bin/activate
echo "Starting Tennis Analysis Server..."
echo "Please keep this window open while using the dashboard."
python app_analysis.py
