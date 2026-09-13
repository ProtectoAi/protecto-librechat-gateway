#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
COMPOSE_FILE="$SCRIPT_DIR/compose.yml"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not available in PATH." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Error: Docker Compose v2 is not available." >&2
    exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: Compose file not found: $COMPOSE_FILE" >&2
    exit 1
fi

echo "Stopping Protecto LibreChat Gateway..."
docker compose --file "$COMPOSE_FILE" down
echo "Protecto LibreChat Gateway stopped. The image and log volume were preserved."
