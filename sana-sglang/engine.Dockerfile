FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir \
    "diffusers>=0.36" \
    "transformers>=4.53" \
    accelerate \
    sentencepiece \
    protobuf \
    safetensors \
    fastapi \
    "uvicorn[standard]" \
    pillow \
    python-multipart

WORKDIR /app
COPY engine_diffusers.py .

ENV HF_HOME=/data/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    SANA_PORT=30000 \
    PYTHONUNBUFFERED=1

EXPOSE 30000
# idle = model intentionally unloaded via Clear VRAM; both are healthy states.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=10 \
  CMD python -c "import urllib.request,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:30000/health',timeout=8)); \
exit(0 if r['status']=='ok' and r['state'] in ('ready','idle') else 1)"

CMD ["python", "engine_diffusers.py"]
