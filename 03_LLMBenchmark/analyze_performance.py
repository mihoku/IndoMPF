import pandas as pd
import argparse

def analyze_by_dimension(df):
    """
    Analyzes model performance grouped by table dimension category.
    """
    print("--- Analysis by Table Dimension ---")
    
    # Group by the dimension category and calculate counts and sums of the overall score
    # 'count' gives the total questions, 'sum' gives the number of correct ones (since score is 1)
    analysis = df.groupby('dimension_category')['overall_score'].agg(['count', 'sum']).reset_index()
    analysis = analysis.rename(columns={'count': 'total_questions', 'sum': 'correct_answers'})
    
    # Calculate wrong answers and percentages
    analysis['wrong_answers'] = analysis['total_questions'] - analysis['correct_answers']
    analysis['correct_percentage'] = (analysis['correct_answers'] / analysis['total_questions'] * 100).round(2)
    analysis['wrong_percentage'] = (analysis['wrong_answers'] / analysis['total_questions'] * 100).round(2)
    
    print(analysis.to_string(index=False))
    print("-" * 35 + "\n")


def categorize_reasoning_steps(steps):
    """Categorizes the number of reasoning steps into predefined bins."""
    if steps <= 5:
        return "1-5 steps"
    elif 6 <= steps <= 10:
        return "6-10 steps"
    else:
        return "> 10 steps"

def analyze_by_reasoning_steps(df):
    """
    Analyzes model performance grouped by the number of reasoning steps.
    """
    print("--- Analysis by Number of Reasoning Steps ---")
    
    if 'num_reasoning_steps' not in df.columns:
        print("Warning: 'num_reasoning_steps' column not found. Skipping this analysis.")
        return
        
    # Create a new column for the step category
    df['step_category'] = df['num_reasoning_steps'].apply(categorize_reasoning_steps)
    
    # Group by the new step category
    analysis = df.groupby('step_category')['overall_score'].agg(['count', 'sum']).reset_index()
    analysis = analysis.rename(columns={'count': 'total_questions', 'sum': 'correct_answers'})
    
    # Calculate wrong answers and percentages
    analysis['wrong_answers'] = analysis['total_questions'] - analysis['correct_answers']
    analysis['correct_percentage'] = (analysis['correct_answers'] / analysis['total_questions'] * 100).round(2)
    analysis['wrong_percentage'] = (analysis['wrong_answers'] / analysis['total_questions'] * 100).round(2)
    
    # Ensure a logical order for the output
    category_order = ["1-5 steps", "6-10 steps", "> 10 steps"]
    analysis['step_category'] = pd.Categorical(analysis['step_category'], categories=category_order, ordered=True)
    analysis = analysis.sort_values('step_category')
    
    print(analysis.to_string(index=False))
    print("-" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the model evaluation report CSV.")
    parser.add_argument("report_file", help="Path to the evaluation_report.csv file.")
    args = parser.parse_args()

    try:
        # Read the CSV report into a pandas DataFrame
        report_df = pd.read_csv(args.report_file)
    except FileNotFoundError:
        print(f"Error: The file '{args.report_file}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit()

    # Run both analyses
    analyze_by_dimension(report_df.copy())
    analyze_by_reasoning_steps(report_df.copy())
