import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag import LinuxRAG
from app.core.config import settings

def test_query(rag, query):
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    result = rag.get_response(query)
    
    print("\n[ANSWER]")
    print(result["answer"])
    
    print("\n[SOURCES]")
    for i, s in enumerate(result["sources"]):
        print(f"{i+1}. {s['source']} (p.{s['page']}) - Score: {s['rerank_score']}")

if __name__ == "__main__":
    if os.name == 'nt':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    rag = LinuxRAG()
    
    queries = [
        # 1. Naming Ambiguity / Command Overlap
        "What is the difference between the 'link' command and the 'ln' command?",
        
        # 2. Obscure/Specific Flag
        "Explain the --indicator-style flag in ls.",
        
        # 3. Complex Natural Language Instruction (Implicit Tar)
        "How do I create a compressed archive of a folder but exclude the '.git' directory?"
    ]
    
    for q in queries:
        test_query(rag, q)
