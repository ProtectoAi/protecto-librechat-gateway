#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
COMPOSE_FILE="$SCRIPT_DIR/compose.yml"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-30}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not available in PATH." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Error: Docker Compose v2 is not available." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required for the startup health check." >&2
    exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: Compose file not found: $COMPOSE_FILE" >&2
    exit 1
fi

echo "Stopping any existing Protecto LibreChat Gateway containers..."
docker compose --file "$COMPOSE_FILE" down

echo "Building and starting Protecto LibreChat Gateway in detached mode..."
docker compose --file "$COMPOSE_FILE" up --detach --build

attempt=1
while [ "$attempt" -le "$HEALTH_RETRIES" ]; do
    if curl --fail --silent --show-error "$HEALTH_URL" >/dev/null 2>&1; then
        echo "Protecto LibreChat Gateway is healthy at $HEALTH_URL"
        docker compose --file "$COMPOSE_FILE" ps
        exit 0
    fi

    sleep 1
    attempt=$((attempt + 1))
done

echo "Error: Protecto LibreChat Gateway did not become healthy at $HEALTH_URL." >&2
docker compose --file "$COMPOSE_FILE" ps >&2
docker compose --file "$COMPOSE_FILE" logs --tail 100 >&2
exit 1
