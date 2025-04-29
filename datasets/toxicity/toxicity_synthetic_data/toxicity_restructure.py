import csv
import re

def extract_toxicity(line):
    # Match both quoted and unquoted text
    match = re.match(r'\s*-\s*(Non-toxic|Toxic):\s*"?(.*?)"?$', line, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, None

def process_toxicity_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content into blocks
    blocks = re.split(r'\n\s*\d+\.', content.strip())
    
    toxicity_pairs = []
    for block in blocks[1:]:  # Skip the first empty split
        lines = block.strip().split('\n')
        current_pair = {'label': lines[0].strip()}
        for line in lines[1:]:
            toxicity_type, text = extract_toxicity(line.strip())
            if toxicity_type and text:
                current_pair[toxicity_type.lower()] = text
        if len(current_pair) == 3:  # Ensure we have label, non-toxic, and toxic
            toxicity_pairs.append(current_pair)

    print(f"Found {len(toxicity_pairs)} complete pairs.")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['label', 'non-toxic', 'toxic']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for pair in toxicity_pairs:
            writer.writerow(pair)

def main():
    input_file = 'data/datasets/toxicity/toxicity_synthetic_data/toxicity_synthetic_500.txt'
    output_file = 'data/datasets/toxicity/toxicity_synthetic_data/toxicity_synthetic_500.csv'

    process_toxicity_file(input_file, output_file)
    print(f"Conversion complete. CSV file saved as {output_file}")

if __name__ == "__main__":
    main()