#!/bin/bash

echo "Building miradb Docker Image"

set -euo pipefail
set -x

PGPASSWORD="miradb"
PORT="5433"
DATABASE="mira_db"
CONTAINER_NAME="mira_db_container"
PG_VERSION="17"
IMAGE="postgres:17"
IMAGE_TAG="miradb.postgres:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMP_FILE="$SCRIPT_DIR/mira_db.sql"

if [[ ! -f "$DUMP_FILE" ]]; then
  echo "Dump file not found: $DUMP_FILE. Please put dump file in $SCRIPT_DIR" >&2
  exit 1
fi

docker pull "$IMAGE"

cid=$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")
if [[ -n "$cid" ]]; then
  echo "Removing existing container: $cid"
  docker rm -f "$cid"
fi

docker run \
  -p "${PORT}:5432" \
  --name "$CONTAINER_NAME" \
  --detach \
  -e POSTGRES_PASSWORD="$PGPASSWORD" \
  -e POSTGRES_DB="$DATABASE" \
  -e PGDATA=/var/lib/postgresql/pgdata \
  --shm-size 1gb \
  "$IMAGE"

until docker exec "$CONTAINER_NAME" pg_isready -U postgres >/dev/null 2>&1; do
  echo "Waiting for Postgres..."
  sleep 1
done

docker exec -i "$CONTAINER_NAME" psql -U postgres -d "$DATABASE" -v ON_ERROR_STOP=1 < "$DUMP_FILE"

echo "Checking database is populated with the following tables:"

docker exec "$CONTAINER_NAME" psql -U postgres -d "$DATABASE" -c \
  "SELECT 'extraction_method' AS t, count(*) FROM extraction_method
   UNION ALL SELECT 'mira_template_models', count(*) FROM mira_template_models
   UNION ALL SELECT 'ode_expressions', count(*) FROM ode_expressions
   UNION ALL SELECT 'text_contents', count(*) FROM text_contents
   UNION ALL SELECT 'text_references', count(*) FROM text_references;"

docker stop "$CONTAINER_NAME"
docker commit "$CONTAINER_NAME" "$IMAGE_TAG"
echo "Built image: $IMAGE_TAG"
