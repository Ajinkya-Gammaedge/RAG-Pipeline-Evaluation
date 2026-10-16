#!/bin/sh
set -e

echo "==> [Entrypoint] Running database migrations..."
alembic upgrade head

if [ "${AUTO_SEED:-true}" = "true" ]; then
    echo "==> [Entrypoint] Checking and seeding corpus..."
    python -m app.ingestion.run_scifact
fi

echo "==> [Entrypoint] Starting application server..."
exec "$@"