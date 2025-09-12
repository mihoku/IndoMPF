import json
from sklearn.model_selection import train_test_split

def parse_golden_dataset(file_path: str):
    """
    Reads the complete JSON file, extracts all questions, and maps them
    to their correct table UID. It combines the table's type_id with the question.

    Args:
        file_path (str): The path to dataset_table_final.json

    Returns:
        list: A flat list of {'question': ..., 'correct_table_uid': ...} dictionaries.
    """
    print(f"Parsing golden dataset from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

    qa_pairs = []
    for entry in data:
        table_info = entry.get("table", {})
        correct_table_uid = table_info.get("uid")
        type_id = table_info.get("type_id", "")

        if not correct_table_uid:
            continue

        for question_obj in entry.get("questions", []):
            question_text = question_obj.get("question")
            if question_text:
                # Combine type_id and question as requested
                combined_question = f"{type_id} {question_text}"
                
                qa_pairs.append({
                    "question": combined_question,
                    "correct_table_uid": correct_table_uid
                })
    
    print(f"Successfully parsed {len(qa_pairs)} question-answer pairs.")
    return qa_pairs

def prepare_datasets(input_file="data/dataset_table_final.json"):
    """
    Parses the main data file and splits it into train, validation, and test sets.
    """
    # Step 1: Parse the complex JSON into a simple question-answer list
    qa_data = parse_golden_dataset(input_file)

    if not qa_data:
        print("Aborting split due to parsing failure.")
        return

    # Step 2: Split the parsed data
    # First split: 80% for training/validation, 20% for testing
    train_val_data, test_data = train_test_split(
        qa_data,
        test_size=0.20,
        random_state=42 # for reproducibility
    )

    # Second split: Split the 80% into training and validation
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=0.20,
        random_state=42
    )

    print("\nDataset split complete:")
    print(f"- Training set size: {len(train_data)}")
    print(f"- Validation set size: {len(val_data)}")
    print(f"- Test set size: {len(test_data)}")

    # Step 3: Save the splits to separate files
    with open('data/train_set.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    with open('data/val_set.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)
    with open('data/test_set.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)

    print("\nDatasets saved to data/train_set.json, data/val_set.json, and data/test_set.json")

# --- Putting It All Together ---
if __name__ == '__main__':
    # would run this single function to parse source file and create all three dataset splits.
    prepare_datasets(input_file="data/dataset_table_final.json")