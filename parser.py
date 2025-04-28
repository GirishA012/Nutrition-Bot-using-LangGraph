from llama_parse import LlamaParse
import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()


LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


# Initialize parser
parser = LlamaParse(
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",  
    verbose=True,
)


pdf_folder = Path(r"C:\Users\47770057\Desktop\NBot\Dataset")
pdf_files = list(pdf_folder.glob("*.pdf"))


docs = parser.load_data(pdf_files)
print(f"Parsed {len(docs)} documents.\n")

print(f"{len(docs)}, {len(pdf_files)}")

print(f"Found {len(pdf_files)} PDF files. Parsed {len(docs)} documents.\n")


for doc, pdf_file in zip(docs, pdf_files):
    print(f"\n--- Document: {pdf_file.name} ---")

    
    if hasattr(doc, 'metadata'):
        print("Metadata:", doc.metadata)
    

