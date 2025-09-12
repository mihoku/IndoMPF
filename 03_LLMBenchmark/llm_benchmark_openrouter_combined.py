# Import necessary libraries
from llama_cpp import Llama
import json
import os
import argparse
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import requests
import random
import copy

load_dotenv() # This loads the variables from .env

OR_API_KEY_DICT = {
    "1": os.getenv("OPEN_ROUTER_API_KEY"),
    "2": os.getenv("OPEN_ROUTER_API_KEY_2"),
    "3": os.getenv("OPEN_ROUTER_API_KEY_3"),
    "4": os.getenv("OPEN_ROUTER_API_KEY_4"),
}

# openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
openrouter_api_url = os.getenv("OPEN_ROUTER_API_URL")

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

def format_example_for_prompt(table_uid, example_question):
    """
    Formats a question and its reasoning into a clean JSON string for the prompt,
    excluding 'validity_status' from the question and its operands.
    """
    # Create a deep copy of the reasoning to modify it safely without affecting the original data
    reasoning_copy = copy.deepcopy(example_question.get("reasoning", {}))

    # Iterate through the steps and operands to remove 'validity_status'
    for step_key, step_details in reasoning_copy.items():
        cleaned_operands = []
        for operand in step_details.get("operands", []):
            if isinstance(operand, dict):
                # Create a copy of the operand and remove the validity_status key if it exists
                cleaned_operand = operand.copy()
                if 'validity_status' in cleaned_operand:
                    del cleaned_operand['validity_status']
                cleaned_operands.append(cleaned_operand)
            else:
                # If it's not a dict (e.g., a string like "#0"), keep it as is
                cleaned_operands.append(operand)
        step_details['operands'] = cleaned_operands

    # Create a clean dictionary with a specific key order for clarity in the prompt
    formatted_example = {
        "table_uid": table_uid,
        "question_uid": example_question.get("uid"),
        "question": example_question.get("question"),
        "answer_type": example_question.get("answer_type"),
        "scale": example_question.get("scale"),
        "answer_sentence": example_question.get("answer_sentence"),
        "answer": example_question.get("answer"),
        "reasoning": reasoning_copy # Use the cleaned reasoning object
    }
    return json.dumps(formatted_example, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    # --- 1. Set up argument parser ---
    parser = argparse.ArgumentParser(description="Generate reasoning for all questions in a dataset.")
    parser.add_argument("--file", required=True, help="The JSON dataset file to process (e.g., train_set_90.json).")
    parser.add_argument("--model", required=True, choices=['Gemma', 'Qwen','Deepseek-Qwen','Deepseek-Llama'], help="The LLM to use ('gemma' or 'sahabat').")
    parser.add_argument("--examples_file", required=True, help="JSON file containing high-quality examples for one-shot prompting.")
    parser.add_argument("--zeroshot", default="n")
    parser.add_argument("--oneshot", default="n")
    parser.add_argument("--api_key", default="1")
    args = parser.parse_args()

    openrouter_api_key = OR_API_KEY_DICT[args.api_key]

    # --- 2. Load the entire dataset ---
    all_data = load_json_data(args.file, "main dataset")
    print(f"Loaded {len(all_data)} table entries from '{args.file}'.")

    model_dictionary = {
        "Gemma":{"model":"google/gemma-2-9b-it", "output_file_zeroshot":"qa_gemma.txt", "output_file_oneshot":"qa_gemma_oneshot.txt"},
        "Qwen":{"model":"qwen/qwen3-8b", "output_file_zeroshot":"qa_qwen.txt", "output_file_oneshot":"qa_qwen_oneshot.txt"},
        "Deepseek-Qwen":{"model":"deepseek/deepseek-r1-0528-qwen3-8b", "output_file_zeroshot":"qa_deepseek_qwen.txt", "output_file_oneshot":"qa_deepseek_qwen_oneshot.txt"},
        "Deepseek-Llama":{"model":"deepseek/deepseek-r1-distill-llama-8b", "output_file_zeroshot":"qa_deepseek_llama.txt", "output_file_oneshot":"qa_deepseek_llama_oneshot.txt"}
    }
    
    model_name = model_dictionary[args.model]["model"]

    examples_data = load_json_data(args.examples_file, "examples dataset")
    print(f"Loaded {len(examples_data)} table entries for examples from '{args.examples_file}'.")
    # --- 3. Create a pool of high-quality examples (once) ---
    valid_examples_pool = []
    for entry in examples_data:
        table_uid = entry.get("table", {}).get("uid")
        if not table_uid:
            continue
        for question in entry.get("questions", []):
            if question.get('validity_status') == "matched facts (program parseable)":
                valid_examples_pool.append({
                    'table_uid': table_uid,
                    'question_data': question
                })
    
    if not valid_examples_pool:
        print("Warning: No valid examples found in the examples file. Proceeding with zero-shot.")
    else:
        print(f"Created a pool of {len(valid_examples_pool)} valid examples for one-shot prompting.")

    # --- 5. Main loop to iterate over EACH TABLE in the dataset ---
    for entry_data in tqdm(all_data, desc="Processing Tables"):
        
        table_info = entry_data.get('table', {})
        uid = table_info.get('uid')
        if not uid:
            print("Skipping entry with no table UID.")
            continue

        print(f"\n--- Starting processing for table UID: {uid} ---")

        # --- 6. Prepare prompt parts for the current table ---
        table_title = table_info.get('title', 'No Title Provided')
        paragraphs_list = entry_data.get('paragraphs', [])
        context_paragraphs = "\n".join([p.get('text', '') for p in paragraphs_list])
        table_list = table_info.get('table', [])
        table_dict = {i: row for i, row in enumerate(table_list)}
        table_content_json = json.dumps(table_dict, indent=2, ensure_ascii=False)

        # --- 7. Prepare for iterative saving for this specific table ---
        output_filename_zeroshot = model_dictionary[args.model]["output_file_zeroshot"]
        output_filename_oneshot = model_dictionary[args.model]["output_file_oneshot"]

        
        if args.zeroshot=="y":
            # --- 3. Read the Prompt Template File (once) ---
            try:
                with open('testing_prompt.txt', 'r', encoding='utf-8') as f:
                    prompt_template_zeroshot = f.read()
            except FileNotFoundError:
                print("Error: 'testing_prompt.txt' not found in the current directory.")
                exit()
            
            # Create the base prompt for this table
            base_prompt_zeroshot = prompt_template_zeroshot.replace("<<Table_Title>>", table_title)
            base_prompt_zeroshot = base_prompt_zeroshot.replace("<<Table_UID>>", uid)
            base_prompt_zeroshot = base_prompt_zeroshot.replace("<<Context_Paragraph>>", context_paragraphs)
            base_prompt_zeroshot = base_prompt_zeroshot.replace("<<Table_Content>>", table_content_json)
            
            try:
                with open(output_filename_zeroshot, 'r', encoding='utf-8') as f:
                    existing_results_zeroshot = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_results_zeroshot = []
            
            processed_uids_zeroshot = {result['question_uid'] for result in existing_results_zeroshot}

        if args.oneshot=="y":
            # --- 3. Read the Prompt Template File (once) ---
            try:
                with open('testing_prompt_shot.txt', 'r', encoding='utf-8') as f:
                    prompt_template_oneshot = f.read()
            except FileNotFoundError:
                print("Error: 'testing_prompt_shot.txt' not found in the current directory.")
                exit()

            # Create the base prompt for this table
            base_prompt_oneshot = prompt_template_oneshot.replace("<<Table_Title>>", table_title)
            base_prompt_oneshot = base_prompt_oneshot.replace("<<Table_UID>>", uid)
            base_prompt_oneshot = base_prompt_oneshot.replace("<<Context_Paragraph>>", context_paragraphs)
            base_prompt_oneshot = base_prompt_oneshot.replace("<<Table_Content>>", table_content_json)

            try:
                with open(output_filename_oneshot, 'r', encoding='utf-8') as f:
                    existing_results_oneshot = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_results_oneshot = []
            processed_uids_oneshot = {result['question_uid'] for result in existing_results_oneshot}
               
        questions_to_process = entry_data.get('questions', [])
        if not questions_to_process:
            print(f"No questions found for table {uid}. Skipping.")
            continue
            
        # --- 8. Inner loop to iterate over EACH QUESTION for the current table ---
        for question in tqdm(questions_to_process, desc=f"Questions for {uid}", leave=False):
            question_uid = question.get('uid')
            question_text = question.get('question', '')

            if not question_text:
                continue
            
            if args.zeroshot=="y":
                if question_uid in processed_uids_zeroshot:
                    continue
                else:
                    # Finalize the prompt for this specific question
                    final_prompt = base_prompt_zeroshot.replace("<<Question_UID>>", question_uid)
                    final_prompt = final_prompt.replace("<<Question>>", question_text)
                    # SETUP THE REQUEST
                    try:
                        response = requests.post(
                            url=openrouter_api_url,
                            headers={
                                "Authorization": f"Bearer {openrouter_api_key}",
                            },
                            data=json.dumps({
                                "model": model_name,
                                "messages": [
                                {"role": "user", "content": final_prompt}
                                ]
                            })
                        )

                        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

                        # 3. PROCESS THE RESPONSE
                        response_json = response.json()
                        answer = response_json['choices'][0]['message']['content']

                        print("✅ Answer received:")
                        print(answer.strip())

                    except requests.exceptions.HTTPError as err:
                        print(f"❌ HTTP Error: {err}")
                        print(f"Response body: {err.response.text}")
                        continue
                    except Exception as e:
                        print(f"❌ An error occurred: {e}")
                        continue

                    generated_text_zeroshot = answer.strip()          

                    # Append the new result
                    existing_results_zeroshot.append({
                        "table_uid": uid,
                        "question_uid": question_uid,
                        "generated_reasoning_text": generated_text_zeroshot.strip()
                    })
                    # new_result_zeroshot = {
                    #     "table_uid": uid,
                    #     "question_uid": question_uid,
                    #     "generated_reasoning_text": generated_text_zeroshot.strip()
                    # }

                    # Save progress after each question
                    try:
                        # with open(output_filename_zeroshot, 'a', encoding='utf-8') as f:
                        #     f.write(json.dumps(new_result_zeroshot, indent=4, ensure_ascii=False) + '\n')
                        with open(output_filename_zeroshot, 'w', encoding='utf-8') as f:
                            json.dump(existing_results_zeroshot, f, indent=4, ensure_ascii=False)
                            print(f"{args.model}output for question {question_uid} using table {uid} saved!")
                    except IOError as e:
                        print(f"\nCRITICAL ERROR: Could not save progress to {output_filename_zeroshot}: {e}. Stopping.")
                        exit()
                
                    time.sleep(1) # Optional delay

            if args.oneshot=="y":
                if question_uid in processed_uids_oneshot:
                    continue
                else:
                    
                    example_question_string = "" # Default to zero-shot
                    if valid_examples_pool:
                        # Filter the pool for examples from the same table, but not the same question
                        table_specific_examples = [
                            ex for ex in valid_examples_pool 
                            if ex['table_uid'] == uid and ex['question_data']['uid'] != question_uid
                        ]
                        
                        if table_specific_examples:
                            random.seed(question_uid) # Use question UID for reproducible selection
                            chosen_example = random.choice(table_specific_examples)
                            example_question_string = format_example_for_prompt(
                                chosen_example['table_uid'], chosen_example['question_data']
                            )

                    # Finalize the prompt for this specific question
                    final_prompt = base_prompt_oneshot.replace("<<Question_UID>>", question_uid)
                    final_prompt = final_prompt.replace("<<Question>>", question_text)
                    final_prompt = final_prompt.replace("<<example_question_reasoning>>", example_question_string)
                    # SETUP THE REQUEST
                    try:
                        response = requests.post(
                            url=openrouter_api_url,
                            headers={
                                "Authorization": f"Bearer {openrouter_api_key}",
                            },
                            data=json.dumps({
                                "model": model_name,
                                "messages": [
                                {"role": "user", "content": final_prompt}
                                ]
                            })
                        )

                        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

                        # 3. PROCESS THE RESPONSE
                        response_json = response.json()
                        answer = response_json['choices'][0]['message']['content']

                        print("✅ Answer received:")
                        print(answer.strip())

                    except requests.exceptions.HTTPError as err:
                        print(f"❌ HTTP Error: {err}")
                        print(f"Response body: {err.response.text}")
                        continue
                    except Exception as e:
                        print(f"❌ An error occurred: {e}")
                        continue

                    generated_text_oneshot = answer.strip()          

                    # Append the new result
                    existing_results_oneshot.append({
                        "table_uid": uid,
                        "question_uid": question_uid,
                        "generated_reasoning_text": generated_text_oneshot.strip()
                    })

                    # Save progress after each question
                    try:
                        with open(output_filename_oneshot, 'w', encoding='utf-8') as f:
                            json.dump(existing_results_oneshot, f, indent=4, ensure_ascii=False)
                            print(f"{args.model} output for question {question_uid} using table {uid} saved!")
                    except IOError as e:
                        print(f"\nCRITICAL ERROR: Could not save progress to {output_filename_oneshot}: {e}. Stopping.")
                        exit()
                
                    time.sleep(1) # Optional delay            


    print("\n--- All tables processed successfully! ---")
