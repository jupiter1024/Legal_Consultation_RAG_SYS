from fpdf import FPDF

class RAGReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, 'RAG System Evaluation: Hallucination & Retrieval Analysis', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(2)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

pdf = RAGReport()
pdf.add_page()

# Case 1: Impossible Pipeline
pdf.chapter_title('Case 1: Impossible Command Pipeline (Architectural Blindness)')
pdf.chapter_body(
    "Question: 'How can I use cp in a pipeline with tar and tee to create a bootable system backup?'\n\n"
    "System Response: Suggested a command where 'cp' reads from a pipe: 'tar ... | tee ... | cp -a ...'.\n\n"
    "Why it Hallucinated: The system treated 'cp' as a stream filter like 'tar -x'. It lacks a 'command ontology' "
    "to distinguish between tools that read stdin and tools that require file arguments. "
    "The LLM brute-forced an invention to satisfy the request for a single pipeline rather than admitting impossibility."
)

# Case 2: False Causal Relationship
pdf.chapter_title('Case 2: False Causal Logic (Pattern-Matching Bias)')
pdf.chapter_body(
    "Question: 'Why does the cp command NOT preserve hard links by default, but the ln command does NOT create hard links to directories?'\n\n"
    "System Response: Invented a causal link between symbolic link handling and hard link preservation that does not exist.\n\n"
    "Why it Hallucinated: The LLM identified two concepts mentioned near each other in the source material and "
    "assumed they were related. It also performed 'semantic slot filling,' replacing technical terms about "
    "symbolic links with 'directories' to match the user's phrasing, fundamentally changing the technical meaning."
)

# Case 3: Fragmented Reference (ls --indicator-style) - COMBINED
pdf.chapter_title('Case 3: Reference Fragmentation & Technical Omission (Combined)')
pdf.chapter_body(
    "Question: 'Explain the --indicator-style flag in ls.'\n\n"
    "System Response: Omitted the 'file-type' option, claimed '-F' and '--indicator-style' should be used together (redundant), "
    "and cited a non-existent man-page section.\n\n"
    "Why it Hallucinated: \n"
    "1. Chunk Fragmentation: The technical documentation was split across chunk boundaries, causing the tail-end "
    "of the flag descriptions (like 'file-type') to be lost.\n"
    "2. Overgeneralization: The LLM assumed two related options could be combined for additive effect.\n"
    "3. Retrieval Pressure: The Bi-encoder retrieved the 'best guess' chunks, but they were incomplete, leading the LLM "
    "to fill gaps with fabricated citations."
)

# Case 4: Knowledge Leakage / Suggestion Hallucination
pdf.chapter_title('Case 4: Internal Knowledge Leakage (Constraint Violation)')
pdf.chapter_body(
    "Questions: 'How do I install Windows 11...' or 'How to exclude .git in tar?'\n\n"
    "System Response: Suggested valid commands (tar --exclude) that WERE NOT in the retrieved context.\n\n"
    "The Failure: The LLM's internal training data (internal knowledge) overrode the 'Answer from context only' "
    "system constraint. It provided 'correct' Linux commands that were not citation-backed from the private store.\n\n"
    "Architectural Mark: Fidelity failure. The system acts as a generic chatbot instead of a grounded assistant."
)

output_path = 'Hallucination_Evaluation_Report.pdf'
pdf.output(output_path)
print(f'Report generated at: {output_path}')
