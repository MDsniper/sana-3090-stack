FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Pinned to the exact set measured on this box (see README §5). Floating
# ranges here drift on every rebuild: transformers had already moved 4.x -> 5.x
# under a ">=4.53" pin. To move a pin deliberately, bump it, rebuild, and
# re-verify a generation before committing.
RUN pip install --no-cache-dir \
    "diffusers==0.40.0" \
    "transformers==5.16.1" \
    "accelerate==1.14.0" \
    "sentencepiece==0.2.2" \
    "protobuf==7.36.0" \
    "safetensors==0.8.0" \
    "fastapi==0.141.1" \
    "uvicorn[standard]==0.52.4" \
    "pillow==11.3.0" \
    "python-multipart==0.0.32"
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
