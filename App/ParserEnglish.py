import os
import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
import hashlib
from pypdf import PdfReader

# -----------------------------
# 1. TEXT CLEANING UTILS
# -----------------------------

def fix_broken_words(text: str) -> str:
    common_fixes = {
        r'Ar\s+t\s*i\s*c\s*l\s*e': 'Article',
        r'Reg\s*u\s*l\s*a\s*t\s*i\s*o\s*n': 'Regulation',
        r'Sec\s*t\s*i\s*o\s*n': 'Section',
        r'Par\s*a\s*g\s*r\s*a\s*p\s*h': 'Paragraph',
        r'Eu\s*r\s*o\s*p\s*e\s*a\s*n': 'European',
        r'Un\s*i\s*o\s*n': 'Union',
        r'Cou\s*n\s*c\s*i\s*l': 'Council',
        r'Par\s*l\s*i\s*a\s*m\s*e\s*n\s*t': 'Parliament',
        r'Dir\s*e\s*c\s*t\s*i\s*v\s*e': 'Directive'
    }
    for pattern, replacement in common_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    def join_chars(match): return match.group(0).replace(" ", "")
    text = re.sub(r'\b[A-Za-z](?:\s[A-Za-z]){2,}\b', join_chars, text)
    text = re.sub(r'\b([a-z])\s([a-z])\b', r'\1\2', text)
    return text

def clean_line(text: str) -> str:
    text = re.sub(r'\d{1,2}\.\d{1,2}\.\d{4}\s+L\s+\d+/\d+.*?(European Union|EN|Official Journal)', '', text, flags=re.IGNORECASE)
    if re.match(r'^\s*\d+\s*$', text): return ""
    return text.strip()

# -----------------------------
# 2. EXTRACTION
# -----------------------------

def extract_with_metadata(path: str) -> List[Dict]:
    reader = PdfReader(path)
    all_lines = []
    line_counts = Counter()
    temp_pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        temp_pages.append(lines)
        for l in lines:
            if len(l) > 10: line_counts[l] += 1
            
    repeated = {line for line, count in line_counts.items() if count > len(reader.pages) * 0.4}
    current_global_line = 1
    for p_idx, page_lines in enumerate(temp_pages):
        page_num = p_idx + 1
        for l_idx, line in enumerate(page_lines):
            if line in repeated: continue
            cleaned = clean_line(line)
            if not cleaned: continue
            all_lines.append({
                "text": fix_broken_words(cleaned),
                "page": page_num,
                "global_line": current_global_line
            })
            current_global_line += 1
    return all_lines

# -----------------------------
# 3. HIERARCHICAL CHUNKER
# -----------------------------

class LegalDocumentChunker:
    def __init__(self):
        self.patterns = {
            "chapter": re.compile(r'^CHAPTER\s+[IVXLCDM\d]+', re.IGNORECASE),
            "section": re.compile(r'^Section\s+\d+', re.IGNORECASE),
            "article": re.compile(r'^Article\s+\d+', re.IGNORECASE),
            "point": re.compile(r'^(\(\d+\)|^\d+\.|\([a-z]\)|\([ivx]+\))', re.IGNORECASE)
        }

    def chunk_document(self, lines: List[Dict]) -> List[Dict]:
        chunks = []
        current_chunk_text = []
        
        # Hierarchy state
        state = {
            "chapter": None,
            "section": None,
            "article": None,
            "page_start": 1,
            "line_start": 1
        }

        for line_data in lines:
            text = line_data["text"]
            
            # Check for hierarchy changes
            found_header = False
            for level in ["chapter", "section", "article"]:
                if self.patterns[level].match(text):
                    if current_chunk_text:
                        self._add_chunk(chunks, current_chunk_text, state)
                        current_chunk_text = []
                    
                    state[level] = text
                    # Reset lower levels
                    if level == "chapter": state["section"] = state["article"] = None
                    if level == "section": state["article"] = None
                    
                    state["page_start"] = line_data["page"]
                    state["line_start"] = line_data["global_line"]
                    found_header = True
                    break
            
            if found_header:
                current_chunk_text.append(text)
                continue

            # Check for new point/paragraph
            if self.patterns["point"].match(text) and current_chunk_text:
                if sum(len(t) for t in current_chunk_text) > 200:
                    self._add_chunk(chunks, current_chunk_text, state)
                    current_chunk_text = []
                    state["page_start"] = line_data["page"]
                    state["line_start"] = line_data["global_line"]

            if not current_chunk_text:
                state["page_start"] = line_data["page"]
                state["line_start"] = line_data["global_line"]

            current_chunk_text.append(text)
            state["page_end"] = line_data["page"]
            state["line_end"] = line_data["global_line"]

        if current_chunk_text:
            self._add_chunk(chunks, current_chunk_text, state)

        return chunks

    def _add_chunk(self, chunks, text_list, state):
        full_text = " ".join(text_list)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        if len(full_text) < 30: return

        # Build breadcrumb
        breadcrumb = []
        if state["chapter"]: breadcrumb.append(state["chapter"])
        if state["section"]: breadcrumb.append(state["section"])
        if state["article"]: breadcrumb.append(state["article"])
        
        path_str = " > ".join(breadcrumb)
        
        chunks.append({
            "text": f"{path_str}\n{full_text}" if path_str else full_text,
            "metadata": {
                "hierarchy": {
                    "chapter": state["chapter"],
                    "section": state["section"],
                    "article": state["article"]
                },
                "page_start": state["page_start"],
                "page_end": state["page_end"],
                "line_start": state["line_start"],
                "line_end": state["line_end"]
            }
        })

# -----------------------------
# 4. EXECUTION
# -----------------------------

def process_document(path, output_dir):
    filename = os.path.basename(path)
    print(f"Processing: {filename}")
    lines = extract_with_metadata(path)
    chunker = LegalDocumentChunker()
    chunks = chunker.chunk_document(lines)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chunks.jsonl")
    with open(output_path, 'a', encoding='utf-8') as f:
        for i, c in enumerate(chunks):
            json.dump({
                "text": c["text"],
                "metadata": {
                    "chunk_id": hashlib.md5(f"{filename}_{i}".encode()).hexdigest(),
                    "source": filename,
                    **c["metadata"],
                    "processed_at": datetime.now().isoformat()
                }
            }, f, ensure_ascii=False)
            f.write('\n')
    print(f"Created {len(chunks)} hierarchical chunks for {filename}")

if __name__ == "__main__":
    base_path = r"C:\Users\sd\Desktop\RAGNLP\Data\PDFS"
    output_dir = r"C:\Users\sd\Desktop\RAGNLP\Data\Processed_Chunks"
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.endswith(".pdf"):
                process_document(os.path.join(base_path, file), output_dir)
