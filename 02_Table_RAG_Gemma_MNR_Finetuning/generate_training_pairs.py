import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse
from sentence_transformers import InputExample

def create_training_pairs(all_docs, questions_data):
    """
    Generates [question, positive_chunk] pairs for MultipleNegativesRankingLoss.

    Args:
        all_docs (list): A list of LangChain Document objects.
        questions_data (list): A list of dicts, e.g., [{"question": ..., "correct_table_uid": ...}].

    Returns:
        A list of InputExample objects ready for training.
    """
    # Step 1: Create an efficient lookup map from table_uid to its document chunks
    table_to_chunks_map = defaultdict(list)
    print("Creating a lookup map for table chunks...")
    for doc in tqdm(all_docs):
        table_uid = doc.metadata.get("table_name")
        if table_uid:
            table_to_chunks_map[table_uid].append(doc.page_content)

    # Step 2 & 3: Iterate through questions and create a training pair for each correct chunk
    training_pairs = []
    print("Generating [question, positive_chunk] pairs...")
    for item in tqdm(questions_data):
        question = item["question"]
        correct_table_uid = item["correct_table_uid"]

        # Find all chunks that belong to the correct table using the map
        correct_chunks = table_to_chunks_map.get(correct_table_uid)

        if correct_chunks:
            # Create a pair for EACH correct chunk
            for chunk_content in correct_chunks:
                # The required format is InputExample(texts=[query, positive_passage])
                training_pairs.append(InputExample(texts=[question, chunk_content]))
        else:
            # This warning helps find mismatches between questions and documents
            print(f"Warning: No document chunks found for table_uid: {correct_table_uid}")
    
    return training_pairs

# --- Example Usage ---
if __name__ == '__main__':

    # --- Command-Line Interface (CLI) Setup ---
    parser = argparse.ArgumentParser(description="Generate training/validation set.")
    
    # Shared arguments
    parser.add_argument("--langchain_docs", type=str, default="data/all_langchain_documents.pkl", help="pickle of langchain docs")
    parser.add_argument("--input_file", type=str, default="data/train_set.json", help="input file: train/val set.")
    parser.add_argument("--output_file", type=str, default="data/mnr_training_data.pkl", help="output pickle file.")

    args = parser.parse_args()

    # Load documents (from the pickle file)
    try:
        with open(args.langchain_docs, "rb") as f:
            all_documents = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {args.langchain_docs} not found. Please create it first.")
        exit()

    # Load training questions
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            train_questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.input_file} not found.")
        exit()
        
    # Generate the training pairs
    mnr_training_data = create_training_pairs(all_documents, train_questions)

    # Save the list using pickle
    save_path = args.output_file # .pkl is the standard pickle extension
    print(f"Saving {len(mnr_training_data)} pairs to {save_path}...")
    with open(save_path, "wb") as f: # 'wb' is important for writing in binary mode
        pickle.dump(mnr_training_data, f)

    print(f"\n✅ Successfully created {len(mnr_training_data)} pairs for MultipleNegativesRankingLoss saved to {args.output_file}.")

