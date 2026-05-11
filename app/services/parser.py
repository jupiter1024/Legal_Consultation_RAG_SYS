"""
app/services/parser.py
----------------------
PDF extraction + recursive chunking service.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any
from datetime import datetime

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


def clean_text(text: str) -> str:
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def is_junk_line(line: str, repeated_lines: set) -> bool:
    line = line.strip()
    if not line:
        return True
    if re.match(r'^\d+$', line) or re.match(r'^page \d+( of \d+)?$', line, re.IGNORECASE):
        return True
    if line in repeated_lines:
        return True
    return False


def extract_content_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Two-pass extraction: identify headers/footers, then extract clean text."""
    pages_data: List[Dict[str, Any]] = []
    line_frequency: Dict[str, int] = {}

    print(f"  Analysing: {os.path.basename(pdf_path)}")

    with pdfplumber.open(pdf_path) as pdf:
        # Pass 1: frequency analysis
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                for line in text.split('\n'):
                    line = line.strip()
                    if line:
                        line_frequency[line] = line_frequency.get(line, 0) + 1

        num_pages = len(pdf.pages)
        repeated_lines = {
            line for line, count in line_frequency.items()
            if count > num_pages * 0.3
        }

        # Pass 2: clean extraction
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if not text or len(text.strip()) < 100:
                continue

            filtered = [
                l.strip() for l in text.split('\n')
                if not is_junk_line(l, repeated_lines)
            ]
            page_text = '\n'.join(filtered)

            if len(page_text) > 50:
                pages_data.append({
                    'page_number': i + 1,
                    'text': clean_text(page_text)
                })

    return pages_data


def chunk_document(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Recursively split pages into overlapping chunks with provenance metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )

    all_chunks: List[Dict[str, Any]] = []

    for page in pages_data:
        text = page['text']
        page_num = page['page_number']

        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunk_id = hashlib.md5(
                f"{page_num}_{i}_{chunk_text[:30]}".encode()
            ).hexdigest()

            all_chunks.append({
                'text': chunk_text,
                'metadata': {
                    'page': page_num,
                    'chunk_index': i,
                    'chunk_id': chunk_id,
                }
            })

    return all_chunks


def process_document(pdf_path: str, output_dir: str) -> int:
    """Full pipeline: Extract → Chunk → Save JSONL. Returns chunk count."""
    filename = os.path.basename(pdf_path)
    print(f"\n[Parser] Processing: {filename}")

    pages_data = extract_content_from_pdf(pdf_path)
    print(f"  Extracted {len(pages_data)} pages with content.")

    chunks = chunk_document(pages_data)
    print(f"  Generated {len(chunks)} chunks.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'chunks.jsonl')

    with open(output_path, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            record = {
                'text': chunk['text'],
                'metadata': {
                    'source': filename,
                    'processed_at': datetime.now().isoformat(),
                    **chunk['metadata'],
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  Saved to {output_path}")
    return len(chunks)
