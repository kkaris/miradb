# miradb app Dockerfile
# Omit tag in the 'FROM' command so Compose can pass it in
FROM miradb.base

# Set so worker and Python output appear promptly in docker logs
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["gunicorn", "-w", "2", "-t", "300", "-b", "0.0.0.0:5000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "--capture-output", "miradb.sources.app:app"]
