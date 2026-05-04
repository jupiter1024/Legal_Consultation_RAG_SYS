import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from typing import List, Dict, Any

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
MONGODB_URI = "mongodb+srv://jupitermoussa50_db_user:1PzUfJeGTmA4a2MA@cluster0.usuv4rs.mongodb.net"
DB_NAME = "LegalRAG"
COLLECTION_NAME = "Chunks"

import certifi

class VectorStore:
    def __init__(self):
        print("Connecting to MongoDB Atlas...")
        self.client = MongoClient(
            MONGODB_URI,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidHostnames=True
        )
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        
        print("Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def check_document_exists(self, filename: str) -> bool:
        """Checks if any chunks from this filename already exist in the database."""
        return self.collection.find_one({"metadata.source": filename}) is not None

    def vectorize_and_upload(self, input_file: str, skip_existing: bool = False):
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found.")
            return

        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        print(f"Starting vectorization of {total_lines} chunks...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            batch = []
            for line in tqdm(f, total=total_lines, desc="Processing"):
                chunk_data = json.loads(line)
                filename = chunk_data["metadata"]["source"]
                
                if skip_existing and self.check_document_exists(filename):
                    print(f"Skipping {filename} - already exists.")
                    break # Skip the rest of this file's chunks in this jsonl if they belong to the same file
                
                text = chunk_data["text"]
                
                embedding = self.model.encode(text).tolist()
                
                document = {
                    "chunk_id": chunk_data["metadata"]["chunk_id"],
                    "text": text,
                    "embedding": embedding,
                    "metadata": chunk_data["metadata"]
                }
                
                batch.append(document)
                
                if len(batch) >= 50:
                    self.collection.insert_many(batch)
                    batch = []
            
            if batch:
                self.collection.insert_many(batch)

        print(f"\nSuccessfully uploaded {total_lines} vectors to MongoDB Atlas!")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(query).tolist()
        
        # MongoDB Atlas Vector Search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", # Ensure this matches the index name in Atlas
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        return results

if __name__ == "__main__":
    # For testing purposes
    vs = VectorStore()
    # vs.vectorize_and_upload(r"C:\Users\sd\Desktop\RAGNLP\Data\Processed_Chunks\chunks.jsonl")
    # print(vs.search("What are the regulations?"))
