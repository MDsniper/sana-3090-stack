FROM python:3.12-slim

RUN pip install --no-cache-dir "gradio>=5" httpx pillow
WORKDIR /app
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
