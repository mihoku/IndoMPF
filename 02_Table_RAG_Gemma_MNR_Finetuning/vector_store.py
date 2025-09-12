import os
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from data_transform import process_all_tables
import argparse
import pickle
from tqdm import tqdm

def create_vector_store(args):

    data = process_all_tables("data/dataset_table_final.json")

    table_docs = []
    for record in data:
        # For each dictionary in our list, create a Document object.
        # The keys 'page_content' and 'metadata' match what LangChain expects.
        doc = Document(
            page_content=record.get('page_content'),
            metadata=record.get('metadata')
        )
        table_docs.append(doc)

    # Now, 'table_docs' is a list of LangChain Document objects, ready for use.
    print(f"\nCreated {len(table_docs)} LangChain Document objects.")
    
    # Inspect the first document to verify:
    if table_docs:
        print("\nExample of the first document:")
        print(table_docs[0])

    if args.mode=="save":
        # Save the list of Document objects to a file
        with open("data/all_langchain_documents.pkl", "wb") as f:
            pickle.dump(table_docs, f)

    else:
        FAISS_INDEX_PATH = f"data/faiss_index_tables/hybrid_documents_{args.embedding_model.replace('/','-')}" 

        # --- 3. Set up the Retriever ---
        # use sentence-transformers to create embeddings (numerical representations of text)
        # and FAISS (a library from Facebook AI) to create a searchable vector store.

        # Load the embedding model
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

        # --- Manual Batch Processing with tqdm ---
        
        batch_size = 32 # Adjust this based on RAM
        vector_store = None # Initialize an empty vector store

        # Wrap the range generator with tqdm for a progress bar
        print("\nEmbedding documents in batches...")
        for i in tqdm(range(0, len(table_docs), batch_size)):
            # Get the current batch of documents
            batch = table_docs[i:i + batch_size]
            
            if vector_store is None:
                # Create the vector store with the first batch
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                # Add subsequent batches to the existing store
                vector_store.add_documents(batch)

        # Save the completed vector store
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"✅ New vector store created and saved to '{FAISS_INDEX_PATH}'.")

if __name__ == '__main__':

    # --- Command-Line Interface (CLI) Setup ---
    parser = argparse.ArgumentParser(description="Generate vector store.")
    
    # Shared arguments
    parser.add_argument("--embedding_model", type=str, default="gemma3.embedding", help="Embedding model")
    parser.add_argument("--mode", type=str, default="store", help="Whether to create vector store or saving langchain docs")

    args = parser.parse_args()

    create_vector_store(args)
