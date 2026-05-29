# Docker deployment

Two images: **Postgres** (needs `mira_db.sql`) and the **miradb app**.

## Prerequisites

- `docker/mira_db.sql` - plain SQL dump available in build context
- Docker and Docker Compose

## Build the database image

The dump is copied into the image at build time and loaded on **first container
start** when the data directory (in the container) is empty 
(`docker-entrypoint-initdb.d`).

Run this code to do a standaline build of the db container, start it and print table
statistics:
```bash
cd docker
./build_db_docker.sh
```

Optional: tag the image with a  for release:
```bash
IMAGE_TAG=miradb.postgres:2026-05 ./build_db_docker.sh
docker push ghcr.io/your-org/miradb-postgres:2026-05
```

## Run the stack

Run the full stack with:
```bash
cd docker
docker compose up --build
```
Once the stack is started, you should be able to go to the landing page and
check the health endpoint:
- App: http://localhost:5000 (redirects to `/explorer`)
- Health: http://localhost:5000/health

Database connection is configured via `MIRADB_DB_*` in `docker-compose.yml`.

## DB dump update

Replace `mira_db.sql` with the new dump, then rebuild the **db** image and 
start with a **fresh** Postgres data volume. The dump is baked into the image 
at build time; init scripts run only when `PGDATA` is empty.

From the `docker/` directory:

```bash
# 1. Rebuild the db image (re-copy mira_db.sql into the image)
docker compose build db

# If the image still looks stale, force a clean rebuild:
# docker compose build --no-cache db

# 2. Remove containers and the old DB volume (with -v), then start the stack
docker compose down -v
docker compose up -d
```

The above commands in one line:
```bash
docker compose build db && docker compose down -v && docker compose up -d
```

Optional: build and verify outside Compose, then run the stack:

```bash
IMAGE_TAG=miradb.postgres:YYYY-MM ./test_build_db_docker.sh
docker compose down -v
docker compose up -d
```

### Get row counts from the database

```bash
docker compose exec db psql -U postgres -d mira_db -c \
  "SELECT 'extraction_method' AS t, count(*) FROM extraction_method
   UNION ALL SELECT 'mira_template_models', count(*) FROM mira_template_models
   UNION ALL SELECT 'ode_expressions', count(*) FROM ode_expressions
   UNION ALL SELECT 'text_contents', count(*) FROM text_contents
   UNION ALL SELECT 'text_references', count(*) FROM text_references;"
```

If counts or the landing page at `/explorer` still reflect the old data, an old volume was reused - 
run `docker compose down -v` again before `up`.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.postgres` | Postgres 17 + dump in `initdb.d` |
| `Dockerfile` | App image (miradb + Gunicorn) |
| `docker-compose.yml` | `db` + `app` on `miradb_net` |
| `build_db_docker.sh` | `docker build` + optional table-count verification |
