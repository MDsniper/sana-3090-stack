#!/usr/bin/env python3
"""SANA web UI: Gradio frontend for the OpenAI-style /v1/images/generations engine API.

Single fixed model (SANA-1.5 4.8B 1024px) served by the sana-engine container.
Optimal settings (1024x1024, 20 steps, cfg 4.5) are fixed defaults; the only
exposed knobs are per-generation ones. Prompt improvement runs on the local
Qwen llama.cpp server (qwen35-opus-rd:8000).
"""
from __future__ import annotations

import base64
import io
import os
import pathlib
import random
import time
import uuid
from datetime import datetime
import json

from PIL import Image
import gradio as gr
import httpx

ENGINE = os.getenv("SANA_ENGINE_URL", "http://sana-engine:30000")
MODEL = os.getenv("SANA_MODEL", "sana-1.5-4.8b")
LLM_URL = os.getenv("SANA_LLM_URL", "http://qwen35-opus-rd:8000/v1/chat/completions")
OUTPUT_DIR = os.getenv("SANA_OUTPUT_DIR", "/data/outputs")
MAX_SEED = 2**32 - 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed optima for SANA-1.5 4.8B 1024px — not exposed as sliders on purpose.
OPTIMAL = {"size": 1024, "steps": 20, "guidance": 4.5}

# Tuned default negative prompt (from SGLang's SANA sampler config).
DEFAULT_NEGATIVE = (
    "low quality, low resolution, blurry, overexposed, underexposed, "
    "distorted, deformed, disfigured, bad anatomy, extra limbs, "
    "watermark, text, signature, ugly, noisy, artifacts"
)

# Platform format presets — all sizes are multiples of 32 (DC-AE compression)
# and stay at/near the model's ~1M-pixel native budget unless marked HD.
FORMATS = {
    "Square 1:1 — Instagram post, profile": (1024, 1024),
    "Portrait 4:5 — Instagram feed": (896, 1120),
    "Story 9:16 — Reels, TikTok, Shorts": (576, 1024),
    "Story HD 9:16 — phone wallpaper": (1088, 1920),
    "Landscape 16:9 — YouTube thumbnail, web video": (1024, 576),
    "Widescreen 16:9 — YouTube 720p-class, desktop hero": (1536, 864),
    "Web banner 21:9 — site header": (1344, 576),
    "Kindle cover 1:1.6 — KDP ebook": (800, 1280),
    "Kindle HD 1:1.6 — KDP ebook, larger": (1280, 2048),
}

# Best practices for SANA-1.5 (Gemma2 text encoder, 1024px native):
# dense concrete visual description, one coherent scene, no negations.
IMPROVER_SYSTEM = (
    "You rewrite short text-to-image prompts into rich, effective prompts for the "
    "SANA 1.5 image model. Rules:\n"
    "- Keep the user's exact subject and intent; never replace or add new subjects.\n"
    "- Expand with: subject details, action/pose, setting, lighting, atmosphere, "
    "camera or medium (e.g. 35mm photo, oil painting), composition, color mood.\n"
    "- Use concrete visual vocabulary a camera or painter could follow.\n"
    "- One coherent scene. No contradictions, no lists of unrelated styles.\n"
    "- NEVER include negative phrasing ('no', 'without', 'avoid') — the image model "
    "has a separate negative prompt.\n"
    "- No requests for text, letters, or watermarks inside the image.\n"
    "- Output 2-3 flowing sentences (30-70 words).\n"
    "- Reply with ONLY the improved prompt: no preamble, no quotes, no explanation."
)

# Official style presets from NVlabs/Sana app/app_sana.py (prompt templating only)
STYLE_LIST = [
    {"name": "(No style)", "prompt": "{prompt}", "negative_prompt": ""},
    {"name": "Cinematic", "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy", "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"},
    {"name": "Photographic", "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed", "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"},
    {"name": "Anime", "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed", "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast"},
    {"name": "Manga", "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style", "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style"},
    {"name": "Digital Art", "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed", "negative_prompt": "photo, photorealistic, realism, ugly"},
    {"name": "Pixel art", "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics", "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"},
    {"name": "Fantasy art", "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy", "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white"},
    {"name": "Neonpunk", "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional", "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"},
    {"name": "3D Model", "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting", "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting"},
]
STYLES = {s["name"]: (s["prompt"], s["negative_prompt"]) for s in STYLE_LIST}
# Fooocus community style pack (github.com/lllyasviel/Fooocus, 79 presets)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fooocus_styles.json")) as _f:
    STYLES.update({s["name"]: (s.get("prompt") or "", s.get("negative_prompt") or "")
                   for s in json.load(_f)})


def apply_style(style: str, prompt: str, negative: str):
    p, n = STYLES.get(style, STYLES["(No style)"])
    if "{prompt}" in p:
        out = p.replace("{prompt}", prompt)
    elif p:  # suffix-style (no placeholder): keep the user prompt up front
        out = f"{prompt}, {p}"
    else:    # negative-only enhancer (e.g. "Fooocus Enhance")
        out = prompt
    return out, (n + " " + negative).strip()


def improve_prompt(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt to improve first.")
    try:
        r = httpx.post(LLM_URL, json={
            "messages": [
                {"role": "system", "content": IMPROVER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            # Qwen3.5 is reasoning-distilled: it thinks first (reasoning_content),
            # then answers (content). no_think switches are ignored by this template,
            # so budget enough tokens for reasoning + the improved prompt.
            "max_tokens": 3000, "temperature": 0.7,
        }, timeout=300)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Prompt improver unavailable ({type(e).__name__}): {e}") from e
    text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    # Defensive: strip think blocks/quotes if the LLM leaks any
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = text.strip('"').strip()
    if not text:
        raise gr.Error("Prompt improver returned an empty response — try again.")
    return text

def generate(prompt, negative_prompt, use_negative, style, fmt, steps, guidance,
             num_images, seed, randomize):
    if randomize or seed is None:
        seed = random.randint(0, MAX_SEED)
    neg = negative_prompt if use_negative else DEFAULT_NEGATIVE
    styled_prompt, styled_neg = apply_style(style, prompt, neg)
    width, height = FORMATS.get(fmt, (OPTIMAL["size"], OPTIMAL["size"]))
    body = {
        "model": MODEL,
        "prompt": styled_prompt,
        "size": f"{width}x{height}",
        "n": int(num_images),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "seed": int(seed),
        "response_format": "b64_json",
    }
    if styled_neg.strip():
        body["negative_prompt"] = styled_neg

    t0 = time.time()
    # Retry connect errors: engine may be restarting/warming (weights reload ~1 min)
    client = httpx.Client(transport=httpx.HTTPTransport(retries=10), timeout=600)
    r = client.post(f"{ENGINE}/v1/images/generations", json=body)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    images = []
    for item in data["data"]:
        raw = base64.b64decode(item["b64_json"])
        images.append((Image.open(io.BytesIO(raw)), f"seed {seed}"))

    day_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)
    for img, _ in images:
        img.save(os.path.join(day_dir, f"{uuid.uuid4().hex}_{seed}.png"))

    info = (f"**SANA-1.5 4.8B** · {width}×{height} · seed `{seed}` · "
            f"{int(steps)} steps · cfg {guidance}"
            + f" · **{elapsed/len(images):.2f}s/image**")
    return images, seed, info

def reset_optimal():
    return (OPTIMAL["steps"], OPTIMAL["guidance"])


def engine_status_text() -> str:
    try:
        s = httpx.get(f"{ENGINE}/v1/engine/status", timeout=10).json()
        return (f"Engine: **{s['state']}** · VRAM allocated {s['cuda_allocated_mb']:.0f} MB · "
                f"reserved {s['cuda_reserved_mb']:.0f} MB")
    except Exception as e:  # noqa: BLE001
        return f"Engine status unavailable ({type(e).__name__})"


def clear_vram():
    r = httpx.post(f"{ENGINE}/v1/engine/unload", timeout=120)
    r.raise_for_status()
    return engine_status_text() + " — model unloaded; next Generate reloads it (~1 min)"


def caption_from_image(path: str) -> str:
    if not path:
        raise gr.Error("Upload an image first.")
    with open(path, "rb") as f:
        r = httpx.post(f"{ENGINE}/v1/images/caption", files={"file": f}, timeout=300)
    r.raise_for_status()
    cap = r.json()["caption"]
    if not cap.strip():
        raise gr.Error("Captioning returned nothing — try another image.")
    return cap


def load_library():
    """Newest-first (path, caption) pairs for every image ever generated."""
    files = sorted(pathlib.Path(OUTPUT_DIR).rglob("*.png"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [(str(p), f"{p.parent.name}/{p.name}") for p in files[:500]]


NO_SELECTION = "_No image selected — click one in the gallery below._"


def refresh_library():
    """Repaint the gallery and mirror its exact contents into state.

    Selection is by gallery index, so the state must be the same list the
    user is looking at; a refresh reorders it (newest first), which is why
    any pending selection is dropped here.
    """
    items = load_library()
    return items, items, None, NO_SELECTION


def select_library_image(items, evt: gr.SelectData):
    """Remember which library image was clicked (does not leave the tab)."""
    if not items or evt.index is None or evt.index >= len(items):
        return None, "_Selection is stale — hit 🔄 Refresh library and try again._"
    path, caption = items[evt.index]
    return path, f"Selected **{caption}** — now click *📷 Use as input image*."


def use_library_image(path):
    """Send the selected library image to the Remix input on the Generate tab."""
    if not path:
        raise gr.Error("Click an image in the library first, then use this button.")
    return path, gr.Tabs(selected="generate"), gr.Accordion(open=True)


def build_ui():
    # theme belongs on launch() from Gradio 6 on; passing it here is ignored.
    with gr.Blocks(title="SANA Image Generation") as demo:
        lib_items = gr.State([])   # exact list currently painted in the Library gallery
        lib_sel = gr.State(None)   # path of the library image the user clicked
        with gr.Tabs() as tabs:
            with gr.Tab("🎨 Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=3):
                        fmt_dd = gr.Dropdown(list(FORMATS), value=list(FORMATS)[0], label="Format",
                                             info="All sizes tuned to the model's native pixel budget")
                        prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt…")
                        with gr.Row():
                            improve_btn = gr.Button("✨ Improve prompt")
                            go = gr.Button("Generate", variant="primary")
                        with gr.Accordion("Negative prompt & style", open=False):
                            use_neg = gr.Checkbox(False, label="Use custom negative prompt "
                                                              "(unchecked = tuned default)")
                            neg = gr.Textbox(value=DEFAULT_NEGATIVE, label="Negative prompt",
                                             visible=False, lines=3)
                            use_neg.change(lambda v: gr.update(visible=v), use_neg, neg)
                            style_dd = gr.Dropdown(list(STYLES), value="(No style)",
                                                   label="Style preset")
                        with gr.Accordion("📷 Remix from image", open=False) as remix_acc:
                            img_in = gr.Image(type="filepath",
                                              label="Upload an image (or send one from 📚 Library)")
                            remix_btn = gr.Button("📷 Describe image → prompt")
                        with gr.Accordion("Advanced options (optimal: 20 steps · cfg 4.5)",
                                          open=False):
                            with gr.Row():
                                steps = gr.Slider(1, 50, value=OPTIMAL["steps"], step=1,
                                                  label="Sampling steps")
                                guidance = gr.Slider(0.0, 15.0, value=OPTIMAL["guidance"],
                                                     step=0.1, label="CFG Guidance scale")
                            optimal_btn = gr.Button("Reset to optimal (20 steps · cfg 4.5)")
                            vram_md = gr.Markdown(engine_status_text())
                            clear_vram_btn = gr.Button("🧹 Clear VRAM (unload model)")
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(label="Results", format="png", height=420, columns=2)
                        info_md = gr.Markdown()
                        with gr.Row():
                            num_images = gr.Slider(1, 4, value=1, step=1, label="Num images")
                            seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                            randomize = gr.Checkbox(True, label="Randomize seed")
            with gr.Tab("📚 Library", id="library"):
                lib_md = gr.Markdown("Every image ever generated, newest first. "
                                     "Click an image, then the ⬇ button to download — "
                                     "or send it to 📷 Remix as an input image.")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh library")
                    use_btn = gr.Button("📷 Use as input image", variant="primary")
                sel_md = gr.Markdown(NO_SELECTION)
                lib_gallery = gr.Gallery(label="Library", height=560, columns=4,
                                         object_fit="contain")
        lib_out = [lib_gallery, lib_items, lib_sel, sel_md]
        improve_btn.click(improve_prompt, prompt, prompt, api_name="improve_prompt")
        optimal_btn.click(reset_optimal, None, [steps, guidance], api_name="reset_optimal")
        go.click(generate,
                 inputs=[prompt, neg, use_neg, style_dd, fmt_dd, steps, guidance,
                         num_images, seed, randomize],
                 outputs=[gallery, seed, info_md],
                 api_name="generate").then(refresh_library, None, lib_out)
        prompt.submit(generate,
                      inputs=[prompt, neg, use_neg, style_dd, fmt_dd, steps, guidance,
                              num_images, seed, randomize],
                      outputs=[gallery, seed, info_md],
                      api_name=False).then(refresh_library, None, lib_out)
        remix_btn.click(caption_from_image, img_in, prompt, api_name="remix")
        clear_vram_btn.click(clear_vram, None, vram_md, api_name="clear_vram")
        demo.load(engine_status_text, None, vram_md)
        refresh_btn.click(refresh_library, None, lib_out, api_name="library")
        # Clicking only records the pick, so the gallery's own preview/download
        # flow still works; the button is what leaves the tab.
        lib_gallery.select(select_library_image, lib_items, [lib_sel, sel_md],
                           api_name=False)
        use_btn.click(use_library_image, lib_sel, [img_in, tabs, remix_acc],
                      api_name="use_library_image")
        demo.load(refresh_library, None, lib_out)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=int(os.getenv("SANA_PORT", "7860")),
                      theme=gr.themes.Soft(), allowed_paths=[OUTPUT_DIR])
