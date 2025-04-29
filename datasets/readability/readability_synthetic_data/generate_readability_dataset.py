import os
import json
import time
import anthropic
from anthropic import InternalServerError
import argparse

# Load questions
def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

# Generate summaries using Anthropic API with retry mechanism
def generate_summaries(question, max_retries=8, retry_delay=10):
    client = anthropic.Anthropic()
    
    def api_call_with_retry(system_prompt, user_content):
        for attempt in range(max_retries):
            try:
                message = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=256,
                    temperature=0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_content}
                    ]
                )
                return message.content[0].text
            except InternalServerError as e:
                if attempt < max_retries - 1:
                    print(f"API overloaded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {max_retries} attempts. Skipping this summary.")
                    return "Error: Unable to generate summary due to server overload."

    # Generate complex summary
    complex_summary = api_call_with_retry(
        "You are an expert academic. Provide a complex, detailed answer using sophisticated vocabulary and grammar.",
        question
    )

    # Generate simple summary without context
    simple_summary_without_context = api_call_with_retry(
        "You are explaining to a young student. Provide a simple explanation using basic vocabulary and grammar.",
        question
    )

    # Generate simple summary based on the complex one
    simple_summary_with_context = api_call_with_retry(
        "You are explaining to a young student. Simplify the following complex explanation, using simpler vocabulary and grammar while retaining the key information. Immediately start with your simplified explanation:",
        f"Complex explanation: {complex_summary}\n\nPlease simplify this explanation:"
    )

    return complex_summary, simple_summary_without_context, simple_summary_with_context

# Write results to JSON
def write_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2)

# Load existing results
def load_existing_results(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as jsonfile:
            return json.load(jsonfile)
    return {}

def main(start_index, end_index, save_interval):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '500_questions.csv')
    output_file = os.path.join(script_dir, 'summaries.json')

    questions = load_questions(input_file)
    results = load_existing_results(output_file)

    for index in range(start_index, min(end_index, len(questions))):
        question = questions[index]
        print(f"Processing: {question}")
        complex_summary, simple_summary_without_context, simple_summary_with_context = generate_summaries(question)
        results[str(index)] = {
            "Question": question,
            "Complex Summary": complex_summary,
            "Simple Summary (Without Context)": simple_summary_without_context,
            "Simple Summary (With Context)": simple_summary_with_context
        }
        
        # Save results at specified intervals
        if (index + 1) % save_interval == 0:
            write_json(output_file, results)
            print(f"Progress saved to {output_file}")

    # Final save
    write_json(output_file, results)
    print(f"All summaries saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries for questions.")
    parser.add_argument("--start", type=int, default=0, help="Starting index (inclusive)")
    parser.add_argument("--end", type=int, default=500, help="Ending index (exclusive)")
    parser.add_argument("--interval", type=int, default=10, help="Save interval")
    args = parser.parse_args()

    main(args.start, args.end, args.interval)