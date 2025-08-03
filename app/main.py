import os, io, base64, torch, cv2, numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis

# ── constants ──────────────────────────────────────────────────────────
# Location of model weights. Override with the MODELS environment variable
# when launching the container if your models live elsewhere.
MODELS = os.getenv("MODELS", "/models")
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# ── load InstantID pipeline once at startup ────────────────────────────
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

controlnet = ControlNetModel.from_pretrained(
    f"{MODELS}/instantid/ControlNetModel",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    f"{MODELS}/sdxl",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
).to(DEVICE)

pipe.load_ip_adapter_instantid(f"{MODELS}/instantid/ip-adapter.bin")

face_analyzer = FaceAnalysis(
    name="antelopev2",
    root=MODELS,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

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

    info = face_analyzer.get(cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))
    if len(info) == 0:
        raise HTTPException(400, "No face detected")
    info = sorted(info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    embedding = info['embedding']
    kps = info['kps']

    image = pipe(
        prompt=req.prompt,
        image_embeds=embedding,
        image=draw_kps(face, kps),
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
    ).images[0]

    buf = io.BytesIO(); image.save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode()}
