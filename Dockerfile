# Runs the whole app (API + front end) in one container, for any container
# host or local use. The public demo runs on Vercel instead (see vercel.json).
# No API key is baked in: visitors bring their own.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY --chown=app backend backend
COPY --chown=app frontend frontend
COPY --chown=app samples samples

USER app
WORKDIR /app/backend
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
