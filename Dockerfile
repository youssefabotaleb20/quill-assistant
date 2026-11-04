# Runs the whole app (API + front end) in one container, e.g. on a free
# Hugging Face Space (Docker SDK). No API key is baked in: visitors bring
# their own, so hosting costs nothing.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Hugging Face runs containers as user 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt reportlab==5.0.1

COPY --chown=app backend backend
COPY --chown=app frontend frontend
COPY --chown=app samples samples
# Build the demo files in the image so the repo for the host holds no binaries.
RUN python samples/make_samples.py

USER app
WORKDIR /app/backend
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
