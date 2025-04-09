import csv
import re

def extract_sentiment(line):
    match = re.match(r'\s*(?:\d+\.\s)?(\w+):\s*"?(.*?)"?$', line)
    if match:
        return match.group(1), match.group(2)
    return None, None

def process_sentiment_file(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # Split the content into blocks (each containing a positive-negative pair)
    blocks = re.split(r'\n\s*\n', content.strip())

    sentiment_pairs = []
    for block in blocks:
        lines = block.split('\n')
        current_pair = {}
        for line in lines:
            sentiment_type, text = extract_sentiment(line.strip())
            if sentiment_type and text:
                current_pair[sentiment_type.lower()] = text
        if len(current_pair) == 2:
            sentiment_pairs.append(current_pair)

    print(f"Found {len(sentiment_pairs)} complete pairs.")

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['positive', 'negative']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for pair in sentiment_pairs:
            writer.writerow(pair)

def main():
    input_file = 'data/datasets/sentiment/sentiment_500_en.txt'
    output_file = 'data/datasets/sentiment/sentiment_500_en.csv'

    process_sentiment_file(input_file, output_file)
    print(f"Conversion complete. CSV file saved as {output_file}")

if __name__ == "__main__":
    main()