
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from app.services.vectorstore import VectorStore

def main():
    vs = VectorStore()
    
    queries = [
        "Which command should I use to see the size of a file in MB and then find where that command is located?",
        "Show me how to list files WITHOUT using the ls command.",
        "How do I use find to locate empty files modified exactly 3 days ago only in the current directory?",
        "What is the difference between a process and a thread in Linux?",
        "Summarize all security-related commands in the book."
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        results = vs.search(q, limit=5)
        for i, r in enumerate(results):
            print(f"  {i+1}. [Score: {r['score']:.4f}] {r['metadata']['source']} (Page {r['metadata']['page']})")
            print(f"     Text: {r['text'][:250]}...")

if __name__ == "__main__":
    main()
