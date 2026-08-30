#!/usr/bin/env python3
"""SANA-1.5 4.8B engine: OpenAI-style /v1/images/generations on diffusers.

Speaks the same request/response surface the UI (and any OpenAI client) uses:
  POST /v1/images/generations {model, prompt, negative_prompt?, size "WxH",
                               n, num_inference_steps, guidance_scale, seed,
                               response_format:"b64_json"}
  GET /v1/models, GET /health
Measured 9.4 s/image @ 1024px/20 steps on a 3090 (vs 32 s via SGLang on this GPU).
"""
from __future__ import annotations

import base64
import gc
import io
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from diffusers import SanaPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_REPO = os.getenv("SANA_MODEL_REPO", "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers")
SERVED_NAME = os.getenv("SANA_SERVED_NAME", "sana-1.5-4.8b")
MAX_SEED = 2**32 - 1
DTYPE = torch.bfloat16

# Same tuned default the SGLang sampler ships; keeps UI behavior engine-agnostic.
DEFAULT_NEGATIVE = (
    "low quality, low resolution, blurry, overexposed, underexposed, "
    "distorted, deformed, disfigured, bad anatomy, extra limbs, "
    "watermark, text, signature, ugly, noisy, artifacts"
)


class Engine:
    def __init__(self):
        self.pipe: SanaPipeline | None = None
        self.lock = threading.Lock()
        self.state = "idle"  # idle | loading | ready | error
        self.detail = ""

    def load(self):
        with self.lock:
            if self.pipe is not None:
                return
            self.state, self.detail = "loading", f"loading {MODEL_REPO}"
            t0 = time.time()
            try:
                pipe = SanaPipeline.from_pretrained(MODEL_REPO, torch_dtype=DTYPE)
                pipe.to(DEVICE)
                pipe.vae.enable_tiling()
                self.pipe = pipe
                self.state, self.detail = "ready", ""
                print(f"[engine] loaded {MODEL_REPO} in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                self.state, self.detail = "error", str(e)
                raise

    @torch.inference_mode()
    def generate(self, *, prompt: str, negative_prompt: str | None, width: int, height: int,
                 steps: int, guidance: float, n: int, seed: int):
        neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE
        with self.lock:
            if self.pipe is None:
                raise HTTPException(503, f"engine not ready ({self.state}: {self.detail})")
            gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
            t0 = time.time()
            out = self.pipe(
                prompt=prompt,
                negative_prompt=neg or None,
                height=int(height), width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                num_images_per_prompt=int(n),
                generator=gen,
            )
            print(f"[engine] {n} image(s) in {time.time()-t0:.1f}s", flush=True)
            return list(out.images)


ENGINE = Engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=ENGINE.load, daemon=True).start()
    yield
    ENGINE.pipe = None
    gc.collect()
    torch.cuda.empty_cache()


app = FastAPI(title="SANA Diffusers Engine", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ImageRequest(BaseModel):
    model: str | None = None
    prompt: str
    negative_prompt: str | None = None
    size: str = Field("1024x1024", pattern=r"^\d{3,4}x\d{3,4}$")
    n: int = Field(1, ge=1, le=4)
    num_inference_steps: int = Field(20, ge=1, le=50)
    guidance_scale: float = Field(4.5, ge=0.0, le=15.0)
    seed: int = Field(0, ge=0, le=MAX_SEED)
    response_format: str = Field("b64_json", pattern="^(b64_json|url)$")


@app.get("/health")
def health():
    return {"status": "ok" if ENGINE.state != "error" else "degraded",
            "state": ENGINE.state, "detail": ENGINE.detail,
            "loaded_model": SERVED_NAME if ENGINE.state == "ready" else None}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{
        "id": SERVED_NAME, "object": "model", "owned_by": "diffusers",
        "task_type": "T2I", "dit_precision": "bf16",
    }]}


@app.post("/v1/images/generations")
def generate(req: ImageRequest):
    try:
        w, h = (int(v) for v in req.size.split("x"))
    except ValueError:
        raise HTTPException(400, f"bad size '{req.size}'") from None
    # DC-AE compresses 32x: dimensions must be multiples of 32 or decode breaks.
    w, h = max(256, round(w / 32) * 32), max(256, round(h / 32) * 32)
    images = ENGINE.generate(prompt=req.prompt, negative_prompt=req.negative_prompt,
                             width=w, height=h, steps=req.num_inference_steps,
                             guidance=req.guidance_scale, n=req.n, seed=req.seed)
    data = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
    return {"created": int(time.time()), "data": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SANA_PORT", "30000")), workers=1)
