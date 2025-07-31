import os, io, base64, torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline

# ── constants ──────────────────────────────────────────────────────────
MODELS = "/models"                      # baked into the Docker image
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# ── load InstantID pipeline once at startup ────────────────────────────
pipe = StableDiffusionXLPipeline.from_pretrained(
    f"{MODELS}/sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None
).to(DEVICE)

# attach InstantID adapters
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
pipe = StableDiffusionXLInstantIDPipeline(**pipe.components).to(DEVICE)

app = FastAPI()

class GenerateReq(BaseModel):
    source_image: str            # base64 PNG/JPEG head-shot
    prompt: str                  # “little firefighter, storybook style”
    width: int = 1024
    height: int = 768
    steps: int = 30

@app.get("/health")
def health():
    return {"status":"healthy","gpu":torch.cuda.is_available()}

@app.post("/generate")
def generate(req: GenerateReq):
    try:
        face = Image.open(io.BytesIO(base64.b64decode(req.source_image))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Bad base64 source_image")

    image = pipe(
        prompt=req.prompt,
        input_id_image=face,          # InstantID conditioning
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps
    ).images[0]

    buf = io.BytesIO(); image.save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode()}
