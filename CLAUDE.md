# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A two-container text-to-image stack: a FastAPI **engine** (`sana-engine`, port 30000) that owns
GPU 1 and serves SANA-1.5 4.8B through an OpenAI-shaped `/v1/images/generations` API, and a
stateless Gradio **UI** (`sana-ui`, port 7860) that talks to it over HTTP. A third, pre-existing
service (`qwen35-opus-rd`, llama.cpp on GPU 0, port 8000) provides the ✨ prompt improver.

`README.md` is the design document — it carries the measured benchmarks, VRAM budget, and the
rationale for every fixed default. Read it before changing model, engine, or preset choices;
the numbers in it were measured on the target box and should not be edited speculatively.

## Repo vs. live deployment — read this first

This git repo (`~/sana-3090-stack`) is a **mirror**, not the deploy directory. Docker Compose runs
from `~/ai-stack/`, which holds its own separate copies of `docker-compose.yml` and `sana-sglang/`
(plus `models/`, the `.env`, and the legacy `sana/` stack that are not in git).

Editing a file here changes nothing until it is copied over:

```bash
cp sana-sglang/*.py sana-sglang/*.json sana-sglang/*.Dockerfile ~/ai-stack/sana-sglang/
cp docker-compose.yml ~/ai-stack/
cd ~/ai-stack && docker compose up -d --build
```

Confirm the two trees agree before and after any change: `diff -rq ~/ai-stack/sana-sglang ./sana-sglang`
(`__pycache__` is the only expected difference).

## Commands

```bash
cd ~/ai-stack

docker compose up -d --build            # rebuild + redeploy everything
docker compose up -d --build sana-ui    # UI only (~seconds; still restarts engine via depends_on)
docker logs -f sana-engine              # engine logs: load time, s/image per request
docker logs -f sana-ui

curl -s localhost:30000/health          # {status, state: idle|loading|ready|error, loaded_model}
curl -s localhost:30000/v1/engine/status  # VRAM allocated/reserved
curl -s -X POST localhost:30000/v1/engine/unload   # frees ~15 GB; next generate lazily reloads

# end-to-end smoke test (bypasses the UI)
curl -s localhost:30000/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox in a snowy forest","size":"1024x1024","n":1,
       "num_inference_steps":20,"guidance_scale":4.5,"seed":42}' | head -c 200
```

There is no test suite, linter, or CI. Verification is manual: hit the engine with the curl above
(expect ~9.4 s/image at 1024×1024/20 steps, ~1.4× slower on the first call after a cold start),
then check the UI at `:7860`. Container rebuild is the only way to pick up a source change — the
Python files are `COPY`d into the images, not bind-mounted.

## Architecture invariants

- **The engine is the only CUDA consumer.** `ui.py` imports no torch and runs on `python:3.12-slim`
  with no CUDA — keep it that way so UI changes rebuild in seconds. Anything needing the GPU
  (captioning, upscaling, a new model) belongs behind a new engine endpoint.
- **The API surface is the contract.** `engine.sglang.Dockerfile` is a drop-in alternative engine
  that speaks the same routes. Do not add engine features by widening the UI's coupling to
  diffusers internals; extend the HTTP API instead.
- **Generation is lock-serialized.** `Engine.lock` guards load, unload, and generate — one pipeline,
  one GPU, one request at a time. `_load_locked()` runs inside `generate()` so a Clear VRAM unload
  is transparently followed by a lazy reload.
- **Every width/height must be a multiple of 32.** The DC-AE VAE compresses 32× and decode breaks
  otherwise. The engine defensively rounds (`round(v/32)*32`, floor 256); the UI's `FORMATS` table
  is pre-snapped, which is why "1080×1920" appears as 1088×1920. New presets must be /32-safe and
  should stay near the ~1 M-pixel native budget unless deliberately marked HD.
- **`idle` is a healthy state.** The engine healthcheck accepts `ready` *or* `idle`, because Clear
  VRAM intentionally leaves the model unloaded. Don't tighten it to `ready` only.
- **BLIP is loaded and freed per caption call** so "Clear VRAM" stays truthful. Keep any future
  auxiliary model on the same load-use-free pattern.

## Working in `ui.py`

- `OPTIMAL` (1024 / 20 steps / cfg 4.5) is deliberately fixed, not sliders on the main panel; steps
  and CFG live in a collapsed Advanced accordion with a Reset button. Don't promote tuning knobs to
  the primary surface.
- Styles come from two sources merged into one `STYLES` dict: 10 NVlabs presets inline, plus 277
  from `fooocus_styles.json`. `apply_style()` handles three shapes — `{prompt}` placeholder,
  suffix-only (prepends the user prompt), and negative-only enhancers. Preserve all three when
  touching it.
- The improver targets a reasoning-distilled Qwen: `enable_thinking:false` / `/no_think` are ignored
  by the served chat template, so the request budgets `max_tokens: 3000` and reads
  `message.content` (llama.cpp splits reasoning into `reasoning_content`). Lowering that budget
  truncates the answer away.
- Generation posts through an `httpx` transport with `retries=10` because the engine may be
  reloading weights (~1 min). Keep the retry when changing that call.
- Images are written to `/data/outputs/<YYYY-MM-DD>/<uuid>_<seed>.png` on the `sana-outputs`
  volume; the Library tab reads that volume, so history survives rebuilds. Gradio serves those
  paths via `allowed_paths=[OUTPUT_DIR]`.
- Library selection is **by gallery index**, so the `lib_items` State must always hold the exact
  list the gallery is painting. `refresh_library()` returns both from one `load_library()` call and
  clears any pending selection, because a refresh reorders the list (newest first) and would
  otherwise leave the index pointing at a different image. Every place that repaints the gallery
  must write the whole `lib_out` group. Clicking only records the pick — the ⬇ preview/download
  flow stays intact — and `📷 Use as input image` is what sets `img_in` and switches tabs.

## Configuration

Env vars (defaults in the Dockerfiles / `docker-compose.yml`): `SANA_MODEL_REPO`,
`SANA_SERVED_NAME`, `SANA_CAPTION_REPO`, `SANA_PORT` on the engine; `SANA_ENGINE_URL`,
`SANA_LLM_URL`, `SANA_MODEL`, `SANA_OUTPUT_DIR`, `SANA_PORT` on the UI. Swapping the model means
changing `SANA_MODEL_REPO` **and** revisiting `OPTIMAL`/`FORMATS` in `ui.py`, since those encode
this checkpoint's measured optima.

GPU assignment is pinned in `docker-compose.yml` via `device_ids` — `"1"` for SANA, `"0"` for the
LLM. HF weights live in the `sana-hf-cache` volume; deleting it forces a multi-GB re-download.

## Dependency locking

Builds are reproducible on three levels, and all three must stay consistent:

| Level | Where |
|---|---|
| Base image, by digest | `FROM ...@sha256:` in each Dockerfile (tag kept alongside for readability) |
| Direct deps, exact | `requirements.engine.txt`, `requirements.ui.txt` |
| Transitive closure | `constraints.engine.txt`, `constraints.ui.txt` (generated) |

These were ranges once and drifted across majors unnoticed — `transformers` 4.x → 5.16.1, and
`gradio` 5 → 6, the latter silently dropping the UI's theme. Never relax a pin back to a range.

To move a dependency: bump `requirements.*`, rebuild, regenerate the matching `constraints.*`
(the command is in each file's header), and re-verify a generation plus a caption before
committing. Changing a base digest means re-checking the constraints too, since the base supplies
part of the environment.

The engine's base is **conda**-based, so most of its `pip freeze` output is
`name @ file:///home/conda/...` build paths and torch is a local `+cu128` build — none of it
installable from PyPI. `constraints.engine.txt` therefore locks only the PyPI-installed closure
(`grep -E '^[A-Za-z0-9._-]+==[^+]+$'`); everything conda provides is fixed by the pinned base
digest instead. Don't paste a raw `pip freeze` into that file — the build will fail on the
`file://` lines.

Note the pip layer invalidates on any pin change, making that rebuild a full reinstall (a few
minutes), not the usual seconds.
