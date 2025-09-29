# embedding_main.py - Separate embedding service
import os
import time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

# --- Configuration ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions to match your existing data
DEVICE = "cpu"  # Use CPU for cost efficiency in Function Compute

# --- Request/Response Models ---
class EmbeddingRequest(BaseModel):
    texts: List[str]
    
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    processing_time: float

# --- Global Model Instance (Singleton Pattern) ---
embedding_model = None

def load_model():
    """Load the embedding model once on startup"""
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        start_time = time.time()
        
        # Set torch to use only CPU
        torch.set_num_threads(1)  # Optimize for Function Compute
        
        embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        
        load_time = time.time() - start_time
        print(f"Embedding model loaded successfully in {load_time:.2f} seconds")
    
    return embedding_model

# --- Lifespan Manager ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup to avoid cold start penalties"""
    try:
        load_model()
        print("Embedding service ready")
        yield
    except Exception as e:
        print(f"FATAL: Failed to load embedding model: {e}")
        raise
    finally:
        print("Embedding service shutting down")

# --- FastAPI App ---
app = FastAPI(
    title="Embedding Service",
    description="High-performance embedding generation service",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the provided texts"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        start_time = time.time()
        
        model = load_model()  # Get the loaded model
        
        # Generate embeddings
        embeddings = model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Optional: normalize for better similarity search
            batch_size=8  # Process in small batches for memory efficiency
        )
        
        # Convert to list of lists (JSON serializable)
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            processing_time=processing_time
        )
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = embedding_model is not None
    return {
        "status": "healthy" if model_loaded else "initializing",
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME
    }

@app.get("/")
def root():
    return {"service": "Embedding Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)