import os
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import CTransformers
from langchain.retrievers import BM25Retriever
from dotenv import load_dotenv
import collections
import argparse
import json
import pickle
from tqdm import tqdm
# from data_transform import generate_docs_with_questions, generate_docs_without_questions

def retriever_setup(embedding_model, k=10):

    FAISS_INDEX_PATH = f"data/faiss_index_tables/hybrid_documents_{embedding_model.replace('/','-')}" 

    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # get vector store
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # The retriever can find the most similar documents to a query
    retriever = vector_store.as_retriever(search_kwargs={"k": k}) # Take the top 10 match

    return retriever, vector_store

def hybrid_retriever_setup(embedding_model, k=10):

    FAISS_INDEX_PATH = f"data/faiss_index_tables/hybrid_documents_{embedding_model.replace('/','-')}" 

    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # get vector store
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # The retriever can find the most similar documents to a query
    retriever = vector_store.as_retriever(search_kwargs={"k": k}) # Take the top 10 match

    print("Loading pre-saved LangChain documents...")
    try:
        with open("data/all_langchain_documents.pkl", "rb") as f:
            all_docs = pickle.load(f)
    except FileNotFoundError:
        print("Error: data/all_langchain_documents.pkl not found.")
        exit()

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    return retriever, vector_store, bm25_retriever

# --- 5. Define the Logic to Check if a Question can be Answered ---
def can_answer_question(question: str):
    """
    Uses RAG to determine if a question can be answered by one of the tables.
    """
    print(f"❓ Query: '{question}'")
    # execute setups
    retriever, vector_store = retriever_setup()

    # Step 1: RETRIEVE the most relevant table description
    retrieved_docs = retriever.invoke(question)
    for i in range(len(retrieved_docs)):
        table_name = retrieved_docs[i].metadata["table_name"]
        content = retrieved_docs[i].page_content
        print(f"{table_name} - {content}")

    # --- 2. Extract the 'table_name' from each document's metadata ---
    table_names = [retrieved_docs[i].metadata["table_name"] for i in range(len(retrieved_docs))]

    # --- 3. Use collections.Counter to count the occurrences ---
    # This creates a dictionary-like object with counts: {'sales_fy2023': 3, 'employee_data': 2, ...}
    counts = collections.Counter(table_names)
    # print(f"Candidate retrieved: {counts}")

    # --- 4. Get the single most common table_name and its count ---
    # .most_common(1) returns a list with one tuple: [('sales_fy2023', 3)]
    most_common_item = counts.most_common(1)

    if most_common_item:
        most_common_table_name = most_common_item[0][0]
        count = most_common_item[0][1]
        print(f"The most common table is: '{most_common_table_name}'")
        print(f"It appeared {count} times.")
    else:
        print("No documents were retrieved or metadata was missing.")

    final_retrieved = vector_store.as_retriever(
        search_kwargs={"filter": {"table_name": most_common_table_name}}
    )

    print(f"Retrieved Table: {most_common_table_name}")

def evaluate_ensemble_retriever(test_set_path: str, output_file_path: str, k: int, embedding_model=str):
    """
    Evaluates the retriever on a test set and saves detailed results.
    """
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    correct_at_1 = 0
    correct_at_5 = 0
    reciprocal_ranks = []
    total_questions = len(test_set)
    detailed_results = []

    faiss_retriever, vector_store, bm25_retriever = hybrid_retriever_setup(embedding_model,k)

    print(f"\nEvaluating retriever on {total_questions} test questions...")
    for item in tqdm(test_set):
        question = item["question"]
        correct_table_uid = item["correct_table_uid"]

        # --- 2. Get Results from Both Retrievers ---
        faiss_results = faiss_retriever.invoke(question)
        bm25_results = bm25_retriever.invoke(question)

        # --- 3. Fuse the Ranks using RRF ---
        fused_scores = collections.defaultdict(float)
        rrf_k = 60 # RRF constant, 60 is a common value

        # Process FAISS results
        for i, doc in enumerate(faiss_results):
            doc_uid = doc.metadata.get("uuid") # Use a unique ID for each doc
            if doc_uid:
                fused_scores[doc_uid] += 1 / (rrf_k + i)

        # Process BM25 results
        for i, doc in enumerate(bm25_results):
            doc_uid = doc.metadata.get("uuid")
            if doc_uid:
                fused_scores[doc_uid] += 1 / (rrf_k + i)

        # --- 4. Re-rank and Format the Output ---
        # Create a dictionary of all unique retrieved documents
        all_retrieved_docs = {doc.metadata.get("uuid"): doc for doc in faiss_results + bm25_results}
        
        # Sort the unique doc UIDs by their fused score
        reranked_uids = sorted(fused_scores.keys(), key=lambda uid: fused_scores[uid], reverse=True)
        
        # Build the final retrieved list
        final_retrieved_list = []
        for i, doc_uid in enumerate(reranked_uids[:k]):
            doc = all_retrieved_docs[doc_uid]
            final_retrieved_list.append({
                "rank": i + 1,
                "uid": doc.metadata.get("table_name"),
                "score": fused_scores[doc_uid], # The score is now the RRF score
                "content": doc.page_content
            })

        detailed_results.append({
            "question": question,
            "correct_table_uid": correct_table_uid,
            "retrieved": final_retrieved_list
        })

        # Calculate metrics
        retrieved_uids = [res["uid"] for res in detailed_results[-1]["retrieved"]]
        if retrieved_uids and retrieved_uids[0] == correct_table_uid:
            correct_at_1 += 1
        if correct_table_uid in retrieved_uids[:5]:
            correct_at_5 += 1
            
        rank = 0
        for i, uid in enumerate(retrieved_uids):
            if uid == correct_table_uid:
                rank = i + 1
                break
        reciprocal_ranks.append(1 / rank if rank > 0 else 0)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=4)
    print(f"\n✅ Detailed evaluation results saved to '{output_file_path}'")

    accuracy_at_1 = correct_at_1 / total_questions
    accuracy_at_5 = correct_at_5 / total_questions
    mrr = sum(reciprocal_ranks) / total_questions

    print("\n--- HYBRID Retrieval Evaluation Results ---")
    print(f"Accuracy@1:               {accuracy_at_1:.4f}")
    print(f"Accuracy@5 (Recall@5):    {accuracy_at_5:.4f}")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print("------------------------------------")

def evaluate_retriever(test_set_path: str, output_file_path: str, k: int, embedding_model=str):
    """
    Evaluates the retriever on a test set and saves detailed results.
    """
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    correct_at_1 = 0
    correct_at_5 = 0
    reciprocal_ranks = []
    total_questions = len(test_set)
    detailed_results = []

    retriever, vector_store = retriever_setup(embedding_model,k)

    print(f"\nEvaluating retriever on {total_questions} test questions...")
    for item in tqdm(test_set):
        question = item["question"]
        correct_table_uid = item["correct_table_uid"]

        results_with_scores = vector_store.similarity_search_with_score(question, k=k)
        
        # Store detailed results with content for triplet generation
        detailed_results.append({
            "question": question,
            "correct_table_uid": correct_table_uid,
            "retrieved": [
                {
                    "rank": i + 1,
                    "uid": doc.metadata.get("table_name"),
                    "score": float(score),
                    "content": doc.page_content
                }
                for i, (doc, score) in enumerate(results_with_scores)
            ]
        })

        # Calculate metrics
        retrieved_uids = [res["uid"] for res in detailed_results[-1]["retrieved"]]
        if retrieved_uids and retrieved_uids[0] == correct_table_uid:
            correct_at_1 += 1
        if correct_table_uid in retrieved_uids[:5]:
            correct_at_5 += 1
            
        rank = 0
        for i, uid in enumerate(retrieved_uids):
            if uid == correct_table_uid:
                rank = i + 1
                break
        reciprocal_ranks.append(1 / rank if rank > 0 else 0)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=4)
    print(f"\n✅ Detailed evaluation results saved to '{output_file_path}'")

    accuracy_at_1 = correct_at_1 / total_questions
    accuracy_at_5 = correct_at_5 / total_questions
    mrr = sum(reciprocal_ranks) / total_questions

    print("\n--- Retrieval Evaluation Results ---")
    print(f"Accuracy@1:               {accuracy_at_1:.4f}")
    print(f"Accuracy@5 (Recall@5):    {accuracy_at_5:.4f}")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print("------------------------------------")

# --- Example of How to Run It ---
# if __name__ == '__main__':
#     evaluate_retriever(
#         test_set_path='data/test_set.json',
#         output_file_path='data/evaluation_results.json'
#     )

def generate_triplets_from_eval(evaluation_file_path: str):
    """
    Generates (anchor, positive, hard_negative) triplets for finetuning
    by reading a pre-computed evaluation results file.

    Args:
        evaluation_file_path (str): Path to the JSON file from evaluate_retriever.
    """
    print(f"Loading evaluation results from {evaluation_file_path}...")
    with open(evaluation_file_path, 'r', encoding='utf-8') as f:
        evaluation_results = json.load(f)

    triplets = []
    print("Generating triplets...")
    for result in tqdm(evaluation_results):
        anchor = result["question"]
        correct_uid = result["correct_table_uid"]
        
        positive_text = None
        negative_text = None

        # Find the positive and hard negative texts from the retrieved list
        for retrieved_doc in result["retrieved"]:
            # The first time we see the correct document, that's our positive sample
            if retrieved_doc["uid"] == correct_uid and positive_text is None:
                positive_text = retrieved_doc["content"]
            
            # The first time we see an incorrect document, that's our hard negative
            elif retrieved_doc["uid"] != correct_uid and negative_text is None:
                negative_text = retrieved_doc["content"]

            # If we've found both, we can stop searching for this question
            if positive_text and negative_text:
                break
        
        # Only add the triplet if we successfully found both a positive and a negative
        if anchor and positive_text and negative_text:
            triplets.append([anchor, positive_text, negative_text])

    # Save the triplets to a file
    output_path = evaluation_file_path.replace('.json', '_triplets.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, indent=4)
    
    print(f"\n✅ Saved {len(triplets)} triplets to {output_path}")
    return triplets

# --- Example of How to Run It ---
# if __name__ == '__main__':
#     # First, would have run the updated evaluator on train set
#     # evaluate_retriever('data/train_set.json', 'data/train_eval_results.json')
#
#     # Now, generate triplets from that output file
#     generate_triplets_from_eval('data/train_eval_results.json')
    
def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool for evaluating a retrieval model and generating finetuning data."
    )
    
    # Create subparsers for the different modes (evaluate, triplets)
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of execution")

    # --- Evaluator Mode ---
    parser_eval = subparsers.add_parser("evaluate", help="Run the retrieval evaluation on a dataset.")
    parser_eval.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input test set JSON file (e.g., data/test_set.json)."
    )
    parser_eval.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the detailed evaluation results JSON file."
    )
    parser_eval.add_argument(
        "--k", 
        type=int, 
        default=10, 
        help="The number of top documents to retrieve for evaluation (default: 10)."
    )
    parser_eval.add_argument(
        "--embedding_model", 
        type=str, 
        default="gemma3.embedding.finetuned", 
        help="Embedding model used."
    )
    parser_eval.add_argument(
        "--similarity", 
        type=str, 
        default="density", 
        help="Similarity measurement to use for retrieval."
    )

    # --- Triplets Generator Mode ---
    parser_triplets = subparsers.add_parser("triplets", help="Generate triplets from a pre-computed evaluation file.")
    parser_triplets.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the detailed evaluation results JSON file (generated by the 'evaluate' mode)."
    )
    
    args = parser.parse_args()

    # Execute the chosen mode
    if args.mode == "evaluate":
        print(f"--- Running in EVALUATE mode ---")
        if args.similarity=="density":
            evaluate_retriever(test_set_path=args.input, output_file_path=args.output, k=args.k, embedding_model=args.embedding_model)
        elif args.similarity=="hybrid":
            evaluate_ensemble_retriever(test_set_path=args.input, output_file_path=args.output, k=args.k, embedding_model=args.embedding_model)
    elif args.mode == "triplets":
        print(f"--- Running in TRIPLETS mode ---")
        generate_triplets_from_eval(evaluation_file_path=args.input)

if __name__ == '__main__':
    main()