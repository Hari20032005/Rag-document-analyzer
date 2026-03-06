#!/usr/bin/env bash
set -euo pipefail

mkdir -p backend/data backend/data/uploads
PYTHONPATH=backend python - <<'PY'
from app.services.jobs import job_repository
print('Metadata DB initialized')
PY
