{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from fastapi import FastAPI, HTTPException\
from pydantic import BaseModel\
from sentence_transformers import SentenceTransformer\
from PIL import Image\
import io\
import base64\
\
app = FastAPI()\
\
model = SentenceTransformer("clip-ViT-B-32")\
\
class EmbedImageRequest(BaseModel):\
    image_base64: str\
    mime_type: str | None = None\
\
@app.get("/")\
def root():\
    return \{"status": "ok"\}\
\
@app.post("/embed-image")\
def embed_image(req: EmbedImageRequest):\
    try:\
        image_bytes = base64.b64decode(req.image_base64)\
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")\
\
        emb = model.encode(img)\
\
        return \{\
            "embedding": emb.tolist(),\
            "vector_size": len(emb)\
        \}\
\
    except Exception as e:\
        raise HTTPException(status_code=400, detail=str(e))}