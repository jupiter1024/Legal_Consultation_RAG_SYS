import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from App.vetorization import VectorStore

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
os.environ["GROQ_API_KEY"] = "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"

LEGAL_SYSTEM_PROMPT = """
You are an expert Legal Assistant AI. Your goal is to provide accurate, professional, and cited information based on the provided legal documents.

RULES:
1. Only use the provided context to answer questions. If the answer is not in the context, say you don't know based on the documents.
2. Always cite the Source, Page Number, and Article/Section where applicable.
3. Maintain a formal and objective tone.
4. If there are conflicting pieces of information in the context, highlight them.
5. Structure your response clearly using bullet points or sections.

Context:
{context}
"""

class LegalRAG:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", LEGAL_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

    def get_response(self, question: str) -> Dict[str, Any]:
        # 1. Retrieve relevant chunks
        results = self.vector_store.search(question, limit=5)
        
        if not results:
            return {
                "answer": "I'm sorry, I couldn't find any relevant information in the legal documents provided.",
                "sources": []
            }

        # 2. Prepare context
        context_parts = []
        sources = []
        for res in results:
            context_parts.append(res["text"])
            meta = res["metadata"]
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page_start", "?"),
                "hierarchy": meta.get("hierarchy", {}),
                "text_snippet": res["text"][:200] + "..."
            })
        
        context_text = "\n\n---\n\n".join(context_parts)

        # 3. Generate response
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": context_text,
            "question": question
        })

        return {
            "answer": response.content,
            "sources": sources
        }

if __name__ == "__main__":
    rag = LegalRAG()
    query = "What is the scope of this regulation?"
    response = rag.get_response(query)
    print(f"Q: {query}")
    print(f"A: {response['answer']}")
    print("\nSources:")
    for s in response['sources']:
        print(f"- {s['source']} (Page {s['page']})")