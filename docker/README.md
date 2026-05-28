# Docker deployment

Two images: **Postgres** (needs `mira_db.sql`) and the **miradb app**.

## Prerequisites

- `docker/mira_db.sql` - plain SQL dump available in build context
- Docker and Docker Compose

## Build the database image

The dump is copied into the image at build time and loaded on **first container start** when the data directory (in the container) is empty (`docker-entrypoint-initdb.d`).

```bash
cd docker
./build_db_docker.sh
```

Optional: tag the image for release:

```bash
IMAGE_TAG=miradb.postgres:2026-05 ./build_db_docker.sh
docker push ghcr.io/your-org/miradb-postgres:2026-05
```

Skip the post-build verification run:

```bash
VERIFY=0 ./build_db_docker.sh
```

## Run the stack

```bash
cd docker
docker compose up --build
```

- App: http://localhost:5000 (redirects to `/explorer`)
- Health: http://localhost:5000/health

Database connection is configured via `MIRADB_DB_*` in `docker-compose.yml` (see `.env.example`).

## DB dump update

1. Replace `mira_db.sql` with the latest dump.
2. Rebuild, optionally with a new image tag: `IMAGE_TAG=miradb.postgres:YYYY-MM ./build_db_docker.sh`
3. On each host, use a **new empty data directory** so init runs again:
   - Compose: `docker compose down -v` then `docker compose up`
   - Or remove the old Postgres volume manually before starting the new image tag

If an old volume is reused, Postgres will **not** re-import the SQL even if the image changed.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.postgres` | Postgres 17 + dump in `initdb.d` |
| `Dockerfile` | App image (miradb + Gunicorn) |
| `docker-compose.yml` | `db` + `app` on `miradb_net` |
| `build_db_docker.sh` | `docker build` + optional table-count verification |
