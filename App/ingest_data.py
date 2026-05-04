import os
from App.ParserEnglish import process_document
from App.vetorization import VectorStore

def ingest_all():
    try:
        base_path = "Data/PDFS"
        processed_dir = "Data/Processed_Chunks"
        
        vs = VectorStore()
        
        if not os.path.exists(base_path):
            print(f"Directory {base_path} not found.")
            return

        for file in os.listdir(base_path):
            if file.endswith(".pdf"):
                print(f"\nChecking: {file}")
                if vs.check_document_exists(file):
                    print(f"Document {file} is already indexed. Skipping.")
                    continue
                
                print(f"Indexing new document: {file}")
                
                # 1. Parse
                temp_chunks_file = os.path.join(processed_dir, "temp_chunks.jsonl")
                if os.path.exists(temp_chunks_file):
                    os.remove(temp_chunks_file)
                
                chunks_file = os.path.join(processed_dir, "chunks.jsonl")
                
                if os.path.exists(chunks_file):
                    os.remove(chunks_file)
                    
                process_document(os.path.join(base_path, file), processed_dir)
                
                # 2. Upload
                vs.vectorize_and_upload(chunks_file, skip_existing=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    ingest_all()
