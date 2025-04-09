import json
import random
import logging

def create_random_pairs(input_file, output_json_file):
    try:
        with open(input_file, 'r') as infile:
            sentences = [line.strip() for line in infile if line.strip()]
        
        # Shuffle the sentences to ensure random pairing
        random.shuffle(sentences)
        
        # Check if the number of sentences is sufficient for 500 pairs
        if len(sentences) < 1000:
            raise ValueError("Not enough sentences to create 500 pairs (1000 sentences required)")
        
        # Create 500 pairs, each sentence used exactly once
        pairs = [(sentences[i], sentences[i+1]) for i in range(0, 1000, 2)]
        
        # Log the created pairs
        logging.debug(f"Created {len(pairs)} random pairs")
        
        # Save the pairs to a JSON file
        with open(output_json_file, 'w') as outfile:
            json.dump(pairs, outfile, indent=4)
        
        logging.info(f"Random pairs saved to {output_json_file}")

    except Exception as e:
        logging.error(f"An error occurred during pair creation: {e}")

def main():
    input_file = 'data/datasets/random/random_sentences_cleaned_replaced_1000.txt'  # The cleaned file
    output_json_file = 'data/datasets/random/random_pairs_500.json'

    create_random_pairs(input_file, output_json_file)

if __name__ == "__main__":
    main()
