
import pdfplumber

def main():
    with pdfplumber.open("NLP_RAG_Project.pdf") as pdf:
        for page in pdf.pages:
            print(page.extract_text())
            print("-" * 20)

if __name__ == "__main__":
    main()
