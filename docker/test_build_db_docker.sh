#!/bin/bash

echo "Building miradb postgres image"

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-miradb.postgres:latest}"
VERIFY="${VERIFY:-1}"
CONTAINER_NAME="mira_db_verify"
DATABASE="mira_db"
PGPASSWORD="miradb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMP_FILE="$SCRIPT_DIR/mira_db.sql"

if [[ ! -f "$DUMP_FILE" ]]; then
  echo "Dump file not found: $DUMP_FILE" >&2
  echo "Place mira_db.sql in $SCRIPT_DIR (or fetch from S3 / git lfs pull) before building." >&2
  exit 1
fi

docker build -f "$SCRIPT_DIR/Dockerfile.postgres" -t "$IMAGE_TAG" "$SCRIPT_DIR"
echo "Built image: $IMAGE_TAG"

if [[ "$VERIFY" != "1" ]]; then
  exit 0
fi

cid=$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")
if [[ -n "$cid" ]]; then
  docker rm -f "$cid"
fi

docker run \
  --name "$CONTAINER_NAME" \
  --detach \
  -e POSTGRES_PASSWORD="$PGPASSWORD" \
  -e POSTGRES_DB="$DATABASE" \
  --shm-size 1gb \
  "$IMAGE_TAG"

until docker exec "$CONTAINER_NAME" pg_isready -U postgres -d "$DATABASE" >/dev/null 2>&1; do
  echo "Waiting for Postgres..."
  sleep 2
done

echo "Checking database is populated with the following tables:"

docker exec "$CONTAINER_NAME" psql -U postgres -d "$DATABASE" -c \
  "SELECT 'extraction_method' AS t, count(*) FROM extraction_method
   UNION ALL SELECT 'mira_template_models', count(*) FROM mira_template_models
   UNION ALL SELECT 'ode_expressions', count(*) FROM ode_expressions
   UNION ALL SELECT 'text_contents', count(*) FROM text_contents
   UNION ALL SELECT 'text_references', count(*) FROM text_references;"

docker rm -f "$CONTAINER_NAME"
echo "Verification passed."
