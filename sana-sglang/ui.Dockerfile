# Base pinned by digest; the tag is kept alongside for readability only.
FROM python:3.12-slim@sha256:a249c9f47e05708dd367f3fe8ada03cf347390fad66fb8b0518c0ef55ae3cb84

# Build inputs are fully locked: base digest above, direct pins in
# requirements.ui.txt, transitive closure in constraints.ui.txt. ">=5" once
# resolved to gradio 6, which moved `theme` off the Blocks constructor and
# left the UI rendering unthemed. Keep them exact.
WORKDIR /app
COPY requirements.ui.txt constraints.ui.txt ./
RUN pip install --no-cache-dir -r requirements.ui.txt -c constraints.ui.txt

COPY ui.py fooocus_styles.json ./

ENV SANA_PORT=7860 \
    SANA_OUTPUT_DIR=/data/outputs \
    SANA_ENGINE_URL=http://sana-engine:30000 \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8)"

CMD ["python", "ui.py"]
