#!/usr/bin/env bash
# Create .env from .env.example if it doesn't already exist. Never overwrites.
set -euo pipefail

if [ ! -f .env ]; then
  cp .env.example .env
  echo ".env created from .env.example — edit it to add OPENAI_API_KEY"
else
  echo ".env already exists — leaving it alone"
fi
