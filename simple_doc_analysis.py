import os
import sys
from PyPDF2 import PdfReader
import re

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def analyze_text(text, title):
    """Analyze the extracted text"""
    print(f"\n{'='*80}")
    print(f"Analysis of {title}")
    print(f"{'='*80}")
    
    # Print length information
    print(f"Text length: {len(text)} characters")
    print(f"Approximate word count: {len(text.split())}")
    
    # Extract key entities
    print("\nKey topics and entities:")
    
    # Mental health disorders
    disorders = re.findall(r'(?:Major\s)?(?:Depressive\sDisorder|Depression|Anxiety|Anxiety\sDisorder|Generalized\sAnxiety\sDisorder|Bipolar\sDisorder|Panic\sDisorder|PTSD|Schizophrenia|OCD)', text, re.IGNORECASE)
    if disorders:
        print(f"Disorders mentioned: {', '.join(list(set([d.strip() for d in disorders])))}")
    
    # Therapeutic approaches
    therapies = re.findall(r'(?:Cognitive[\-\s]Behavioral\sTherapy|CBT|Reminiscence\sTherapy|Psychodynamic\sTherapy|Role[\-\s]Playing|Therapy)', text, re.IGNORECASE)
    if therapies:
        print(f"Therapies mentioned: {', '.join(list(set([t.strip() for t in therapies])))}")
    
    # Technology references
    techs = re.findall(r'(?:Knowledge\sGraph|Large\sLanguage\sModel|LLM|AI|Artificial\sIntelligence|Machine\sLearning|Neural\sNetwork|Neo4j|GPT|RAG|Retrieval[\-\s]Augmented\sGeneration|NLP|Natural\sLanguage\sProcessing|Embedding)', text, re.IGNORECASE)
    if techs:
        print(f"Technologies mentioned: {', '.join(list(set([t.strip() for t in techs])))}")
    
    # Print a preview of the text
    print("\nText preview:")
    preview = text[:1000] + "..." if len(text) > 1000 else text
    print(preview)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python simple_doc_analysis.py <file_path>")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Get file extension and title
    _, ext = os.path.splitext(file_path)
    title = os.path.basename(file_path)
    
    # Extract text based on file type
    text = ""
    if ext.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext.lower() == '.docx':
        text = extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return
    
    # Analyze the extracted text
    if text:
        analyze_text(text, title)
    else:
        print(f"No text could be extracted from {title}")

if __name__ == "__main__":
    main()
