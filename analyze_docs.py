import os
import sys
from importlib.machinery import SourceFileLoader

# Load the module with spaces in the filename
DocumentProcessor_module = SourceFileLoader(
    'DocumentProcessor', 
    os.path.join(os.path.dirname(__file__), 'Document and Image Processing for RAG.py')
).load_module()

DocumentProcessor = DocumentProcessor_module.DocumentProcessor

def analyze_document(file_path):
    """Analyze a document and print key information"""
    print(f"\nAnalyzing document: {os.path.basename(file_path)}")
    print("-" * 80)
    
    # Create document processor with minimal settings
    processor = DocumentProcessor(use_embeddings=False)
    
    # Process the document
    success = processor.process_file(file_path)
    
    if success:
        # Get document summary
        summary = processor.get_document_summary()
        print("\nDocument Summary:")
        for key, value in summary.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(value[:5])}{'...' if len(value) > 5 else ''}")
            else:
                print(f"{key}: {value}")
        
        # Get potential memories
        print("\nPotential Key Information:")
        memories = processor.get_potential_memories(10)
        for i, memory in enumerate(memories):
            print(f"\n{i+1}. {memory['text']}")
            print(f"   Topic: {memory['topic']}, Timeframe: {memory['timeframe']}")
            if memory.get('people'):
                print(f"   People: {', '.join(memory['people'])}")
    else:
        print("Failed to process document.")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_docs.py <document_path>")
        sys.exit(1)
    
    # Get file path from command line
    file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Analyze the document
    analyze_document(file_path)
