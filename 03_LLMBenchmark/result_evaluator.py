import json
import re
import argparse
from tqdm import tqdm
import csv

# --- Helper Functions ---

def extract_and_parse_json(raw_string: str) -> dict or list:
    """
    Robustly extracts and parses a JSON object from a string that may be
    incomplete or surrounded by other text and code fences.
    """
    if not isinstance(raw_string, str):
        return {}
    match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```', raw_string, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_brace = raw_string.find('{')
        start_bracket = raw_string.find('[')
        if start_brace == -1 and start_bracket == -1: return {}
        start_pos = min(start_brace if start_brace != -1 else float('inf'),
                        start_bracket if start_bracket != -1 else float('inf'))
        end_brace = raw_string.rfind('}')
        end_bracket = raw_string.rfind(']')
        end_pos = max(end_brace, end_bracket)
        if end_pos == -1 or end_pos < start_pos:
            json_str = raw_string[start_pos:]
        else:
            json_str = raw_string[start_pos : end_pos + 1]
    json_str_no_comments = re.sub(r'^\s*(//|#).*$', '', json_str, flags=re.MULTILINE)
    cleaned_str = json_str_no_comments.strip()
    if cleaned_str.endswith(','): cleaned_str = cleaned_str[:-1]
    if len(re.findall(r'(?<!\\)"', cleaned_str)) % 2 != 0: cleaned_str += '"'
    stack = []
    in_string = False
    for char in cleaned_str:
        if char == '"': in_string = not in_string
        if not in_string:
            if char in ['{', '[']: stack.append(char)
            elif char == '}':
                if stack and stack[-1] == '{': stack.pop()
            elif char == ']':
                if stack and stack[-1] == '[': stack.pop()
    for open_char in reversed(stack):
        if open_char == '{': cleaned_str += '}'
        elif open_char == '[': cleaned_str += ']'
    try:
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        return {}

def load_json_data(file_path, description="data"):
    """Helper function to load data from a JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The data file '{file_path}' was not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        exit()

def extract_program_from_json(full_json: dict) -> list:
    """
    Takes a parsed JSON object, extracts the reasoning part,
    and converts it into a flat list of program tokens.
    """
    try:
        reasoning_data = full_json.get("reasoning", {})
        
        # --- FIX: Handle cases where the LLM returns a list for 'reasoning' ---
        if not isinstance(reasoning_data, dict):
            # If reasoning is not a dictionary, it's unparseable by this logic.
            return []

        if not reasoning_data: return []

        tokens = ['GO']
        sorted_steps = sorted(reasoning_data.items(), key=lambda item: int(item[0][1:]))
        for key, step in sorted_steps:
            tokens.append(step['operation'])
            tokens.append('(')
            operands = step.get('operands', [])
            if not operands:
                tokens.append(')')
                continue
            for op in operands:
                if isinstance(op, str):
                    tokens.append(op)
                elif isinstance(op, dict):
                    source = op.get('source', '')
                    fact = str(op.get('fact', ''))
                    if source in ['table', 'constant']:
                        tokens.append(fact)
                    elif '#' in source:
                        tokens.append(source)
                    else:
                        raise ValueError(f"Unknown operand source: {source}")
                tokens.append(',')
            if tokens[-1] == ',': tokens.pop()
            tokens.append(')')
        tokens.append('EOS')
        return tokens
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        # Gracefully handle any other parsing errors
        return []

def normalize_answer(answer_list):
    """Normalizes an answer list for fair comparison."""
    if not isinstance(answer_list, list):
        return []
    return sorted([str(item).strip() for item in answer_list])

def get_dimension_category(num_rows, num_cols):
    """Determines the dimension category based on the longer of rows or columns."""
    max_dim = max(num_rows, num_cols)
    if max_dim < 30:
        return "< 30 dimension"
    elif 30 <= max_dim <= 50:
        return "30-50 dimension"
    elif 50 < max_dim <= 80:
        return "50-80 dimension"
    elif 80 < max_dim <= 100:
        return "80-100 dimension"
    else:
        return "> 100 dimension"

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM-generated reasoning against a ground truth test set.")
    parser.add_argument("test_set_file", help="Path to the ground truth JSON dataset (e.g., test_set.json).")
    parser.add_argument("llm_output_file", help="Path to the JSON file containing the LLM's output.")
    parser.add_argument("report_file", help="Path to save the detailed CSV report (e.g., evaluation_report.csv).")
    args = parser.parse_args()

    print("Loading datasets...")
    test_set_data = load_json_data(args.test_set_file)
    llm_output_data = load_json_data(args.llm_output_file)
    print("Datasets loaded successfully.")

    llm_output_map = {
        (item['table_uid'], item['question_uid']): item['generated_reasoning_text']
        for item in llm_output_data
    }

    em_score, correct_answer_score, overall_score, total_questions_evaluated = 0, 0, 0, 0
    detailed_results = []

    print("\nStarting evaluation...")
    for entry in tqdm(test_set_data, desc="Evaluating Tables"):
        table_uid = entry.get("table", {}).get("uid")
        table_content = entry.get("table", {}).get("table", [])
        
        num_rows = len(table_content)
        num_cols = len(table_content[0]) if num_rows > 0 else 0
        dimension_category = get_dimension_category(num_rows, num_cols)

        for ground_truth_question in entry.get("questions", []):
            question_uid = ground_truth_question.get("uid")
            
            num_reasoning_steps = len(ground_truth_question.get("reasoning", {}))

            llm_output_text = llm_output_map.get((table_uid, question_uid))

            individual_em_score = 0
            individual_ca_score = 0
            individual_overall_score = 0
            total_questions_evaluated += 1

            if llm_output_text:
                llm_parsed_json = extract_and_parse_json(llm_output_text)
                
                llm_question_data = None
                if isinstance(llm_parsed_json, list):
                    if llm_parsed_json:
                        llm_question_data = llm_parsed_json[0] if isinstance(llm_parsed_json[0], dict) else {}
                elif isinstance(llm_parsed_json, dict):
                    llm_question_data = llm_parsed_json
                
                if llm_question_data:
                    ground_truth_program = extract_program_from_json(ground_truth_question)
                    llm_program = extract_program_from_json(llm_question_data)
                    
                    if ground_truth_program and llm_program and ground_truth_program == llm_program:
                        em_score += 1
                        individual_em_score = 1
                        correct_answer_score += 1
                        individual_ca_score = 1
                    else:
                        ground_truth_answer = normalize_answer(ground_truth_question.get("answer", []))
                        llm_answer = normalize_answer(llm_question_data.get("answer", []))
                        if ground_truth_answer and llm_answer and ground_truth_answer == llm_answer:
                            correct_answer_score += 1
                            individual_ca_score = 1
                
                # Calculate the overall score
                if individual_em_score == 1 or individual_ca_score == 1:
                    individual_overall_score = 1
                    overall_score +=1

            detailed_results.append({
                'table_uid': table_uid,
                'question_uid': question_uid,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'num_reasoning_steps': num_reasoning_steps,
                'dimension_category': dimension_category,
                'em_score': individual_em_score,
                'correct_answer_score': individual_ca_score,
                'overall_score': individual_overall_score
            })

    # Write the detailed results to a CSV file
    try:
        with open(args.report_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'table_uid', 'question_uid', 'num_rows', 'num_cols', 
                'num_reasoning_steps', 'dimension_category', 'em_score', 
                'correct_answer_score', 'overall_score'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
        print(f"\nDetailed report saved to '{args.report_file}'.")
    except IOError:
        print(f"\nError: Could not write the report to '{args.report_file}'.")

    # Print the final summary results
    print("\n--- Evaluation Complete ---")
    if total_questions_evaluated > 0:
        em_percentage = (em_score / total_questions_evaluated) * 100
        ca_percentage = (correct_answer_score / total_questions_evaluated) * 100
        overall_percentage = (overall_score / total_questions_evaluated) * 100

        print(f"Total Questions Evaluated: {total_questions_evaluated}")
        print(f"Correct Answer Score: {correct_answer_score}/{total_questions_evaluated} ({ca_percentage:.2f}%)")
        print(f"Program Exact Match (EM) Score: {em_score}/{total_questions_evaluated} ({em_percentage:.2f}%)")
        print(f"Overall Score (EM or CA is correct): {overall_score}/{total_questions_evaluated} ({overall_percentage:.2f}%)")
    else:
        print("No matching questions were found between the test set and the LLM output file.")
