# Base pinned by digest; the tag is kept alongside for readability only.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385

# Build inputs are fully locked: base digest above, direct pins in
# requirements.engine.txt, transitive closure in constraints.engine.txt.
# These were ranges once and drifted a major version unnoticed (transformers
# 4.x -> 5.x), so a rebuild silently changed what shipped. Keep them exact.
WORKDIR /app
COPY requirements.engine.txt constraints.engine.txt ./
RUN pip install --no-cache-dir -r requirements.engine.txt -c constraints.engine.txt

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
