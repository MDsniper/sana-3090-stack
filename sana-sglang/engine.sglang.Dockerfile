# OPTIONAL: SGLang engine (OpenAI-compatible serving stack).
# Kept for opt-in. Measured on this 3090: 32 s/image vs 9.4 s/image on diffusers
# for the same SANA-1.5 4.8B — SGLang's wins are on datacenter GPUs/batch serving.
# To use: point the compose sana-engine service at this file and add ipc: host,
# shm_size: 32gb. Image quality is identical (CLIP-verified).
FROM lmsysorg/sglang:dev-cu12

# Diffusion extras (sglang.multimodal_gen) are already installed in this image.
ENV HF_HOME=/data/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1

EXPOSE 30000
HEALTHCHECK --interval=30s --timeout=10s --start-period=3600s --retries=20 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=8)"

# Everything resident on GPU 1 (DiT ~9.7GB + Gemma2 ~5.2GB + DC-AE ~1.2GB fits 24GB);
# the auto policy streams encoder/VAE from RAM, which is slower per request.
CMD ["sglang", "serve", \
     "--model-path", "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers", \
     "--served-model-name", "sana-1.5-4.8b", \
     "--component-residency", "text_encoder=resident", "vae=resident", \
     "--host", "0.0.0.0", \
     "--port", "30000"]
