from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64

app = FastAPI()

model = SentenceTransformer("clip-ViT-B-32")

class EmbedImageRequest(BaseModel):
    image_base64: str
    mime_type: str | None = None

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/embed-image")
def embed_image(req: EmbedImageRequest):
    try:
        image_bytes = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        emb = model.encode(img)

        return {
            "embedding": emb.tolist(),
            "vector_size": len(emb),
            "model": "clip-ViT-B-32"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

