# Docker deployment

There are two images: `Dockerfile.postgres` holding the database and 
`Dockerfile` for the app. The database image is built with the SQL dump 
copied in and loaded on first start.

## Prerequisites

- `mira_db.sql` - plain SQL dump available in build context in this directory.
- Docker and Docker Compose

## Build the database image

The dump, `mira_db.sql` is copied into the image at build time and loaded on 
first container start.

A script, `test_build_db_docker.sh`, has been made to do a standalone build of 
the db container, start it, print table statistics and then stop the container.
```bash
cd docker
./test_build_db_docker.sh
```

or, just build the image without printing the statistics:
```bash
cd docker
VERIFY=0 ./test_build_db_docker.sh
```

Optional: tag the database image with a version tag for release:
```bash
IMAGE_TAG=miradb.postgres:2026-05 ./test_build_db_docker.sh
```

## Run the stack

To build and run the full stack from the `docker` directory:
```bash
cd docker
docker compose up --build
```
Once the stack is started, you can go to the landing page and the health 
endpoint:
- App: http://localhost:8003 (redirects to `/explorer`)
- Health: http://localhost:8003/health

The database connection is configured via `MIRADB_DB_*` in `docker-compose.yml`.

## DB dump update

Replace `mira_db.sql` with the new dump, then rebuild the database image. From
the `docker/` directory:

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
./test_build_db_docker.sh && docker compose down -v && docker compose up -d
```

### Get row counts from the database

The following SQL query can be used to verify the row counts in the tables 
after the new dump is loaded.

```bash
docker compose exec db psql -U postgres -d mira_db -c \
  "SELECT 'extraction_method' AS t, count(*) FROM extraction_method
   UNION ALL SELECT 'mira_template_models', count(*) FROM mira_template_models
   UNION ALL SELECT 'ode_expressions', count(*) FROM ode_expressions
   UNION ALL SELECT 'text_contents', count(*) FROM text_contents
   UNION ALL SELECT 'text_references', count(*) FROM text_references;"
```

If counts or the landing page at `/explorer` still reflect the old data, an old 
volume was reused - run `docker compose down -v` again before `up`.

## Files in this directory

| File | Purpose                                                         |
|------|-----------------------------------------------------------------|
| `Dockerfile.postgres` | Postgres 17 + dump in `initdb.d`                                |
| `Dockerfile` | App image (miradb + Gunicorn)                                   |
| `docker-compose.yml` | `db` + `app` on `miradb_net`                                    |
| `test_build_db_docker.sh` | `docker build` for DB image + optional table-count verification |
