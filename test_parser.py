import os
import sys
import pdfplumber
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import config
from src import data_loader

def test_complex_table_parsing():
    """
    Loads all documents and prints a sample of sentences generated from the complex table.
    """
    print("--- Running parser test for complex table ---")
    documents = data_loader.load_all_documents()
    
    # 카테고리별로 문서를 필터링하고 샘플을 출력
    categories_to_check = ["남군", "여군", "남자군무원", "여자군무원"]
    all_found = True

    for category in categories_to_check:
        print(f"\n--- Checking category: {category} ---")
        
        category_docs = [
            doc for doc in documents 
            if doc.metadata.get("category") == category
        ]
        
        if not category_docs:
            print(f"[FAIL] No documents found for category '{category}'.")
            all_found = False
            continue

        print(f"[SUCCESS] Found {len(category_docs)} documents for category '{category}'.")
        print("--- Sample sentences: ---")
        
        # Print first 2 and last 2 samples
        sample_docs = category_docs[:2] + category_docs[-2:]
        for doc in sample_docs:
            print(f"- {doc.page_content}")

    print("\n--- Test Summary ---")
    if all_found:
        print("[PASS] All categories were found and parsed.")
    else:
        print("[FAIL] Some categories were missing. Please check the parsing logic.")


if __name__ == "__main__":
    test_complex_table_parsing()
