import json
import uuid
import nltk
from nltk.tokenize import sent_tokenize

# nltk.download('punkt_tab')

import json

def generate_hybrid_docs(entry: dict, chunk_size: int = 5):
    """
    Generates specialized documents for a SINGLE table entry.
    (This function is now called by the main processing function).
    """
    documents = []
    # --- The ONLY line changed in this function ---
    table_info = entry.get("table", {}) # Changed "table_info" to "table"
    # --- End of change ---

    table_uid = table_info.get("uid")
    if not table_uid:
        return []

    title = table_info.get("title", "")
    table_type = table_info.get("type_id", "")
    paragraphs_text = "\n".join([p.get("text", "") for p in entry.get("paragraphs", [])])

    # 1. Create the single "Conceptual Summary" document
    summary_content = f"Table Title: {title}\nTable Type: {table_type}\n\nContext:\n{paragraphs_text}"
    documents.append({
        "page_content": summary_content,
        "metadata": {"table_name": table_uid, "chunk_type": "summary", "uuid":str(uuid.uuid4())}
    })

    # 2. Process and create "Data Chunks"
    table_data = table_info.get("table", [])
    if len(table_data) < 2:
        return documents

    headers = table_data[0]
    data_rows = table_data[1:]
    flattened_rows = [", ".join(f"{h}='{v}'" for h, v in zip(headers, row)) for row in data_rows]

    for i in range(0, len(flattened_rows), chunk_size):
        row_batch = flattened_rows[i:i + chunk_size]
        rows_content = "\n".join([f"- {row}" for row in row_batch])
        data_chunk_content = f"Table Title: {title}\nTable Type: {table_type}\n\nTable Data Rows:\n{rows_content}"
        
        documents.append({
            "page_content": data_chunk_content,
            "metadata": {"table_name": table_uid, "chunk_type": "data_chunk", "uuid":str(uuid.uuid4()), "rows": f"{i+1}-{i+len(row_batch)}"}
        })

    return documents


def process_all_tables(file_path: str):
    """
    Reads a JSON file containing a list of table entries and generates
    all documents for all tables.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # data is now a list of entries
        print(f"Successfully read {len(data)} table entries from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

    all_docs = []
    # Loop through each entry in the list
    for entry in data:
        # Generate the documents for the current entry
        docs_for_one_table = generate_hybrid_docs(entry, chunk_size=3)
        # Add them to our main list
        all_docs.extend(docs_for_one_table)
    
    return all_docs