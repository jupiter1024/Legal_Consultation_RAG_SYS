import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from App.ParserEnglish import process_document
from App.vetorization import VectorStore
from App.RAG import LegalRAG

app = FastAPI(title="Legal RAG Bot API")

# Directories
UPLOAD_DIR = "Data/PDFS"
PROCESSED_DIR = "Data/Processed_Chunks"
STATIC_DIR = "App/static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize RAG
rag_engine = LegalRAG()

class ChatRequest(BaseModel):
    message: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Parse PDF
        process_document(file_path, PROCESSED_DIR)
        
        # 2. Vectorize and Upload
        # We use a fresh scan of the output directory for vectorization
        chunks_file = os.path.join(PROCESSED_DIR, "chunks.jsonl")
        rag_engine.vector_store.vectorize_and_upload(chunks_file)
        
        return {"message": f"Successfully processed and vectorized {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import traceback

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = rag_engine.get_response(request.message)
        return response
    except Exception as e:
        print("CHAT ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Serve Frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
