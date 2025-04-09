import re
import logging

# Configure logging
logging.basicConfig(filename='clean_file.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_sentence(sentence):
    # Log the sentence before cleaning
    logging.debug(f"Original sentence: '{sentence}'")
    
    # Replace all double quotes with single quotes
    sentence = sentence.replace('"', "'")
    
    # This regex will remove a leading number followed by a period and optional spaces
    cleaned_sentence = re.sub(r'^\d+\.\s+', '', sentence)
    
    # Log the cleaned sentence
    logging.debug(f"Cleaned and replaced sentence: '{cleaned_sentence}'")
    
    return cleaned_sentence

def clean_file(input_file, output_file):
    processed_sentences = set()  # To track unique sentences

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Strip the line of leading/trailing whitespace and log the original line
                line = line.strip()
                logging.debug(f"Processing line: '{line}'")

                # Clean the sentence
                cleaned_line = clean_sentence(line)
                
                # Write only unique, non-empty cleaned lines
                if cleaned_line and cleaned_line not in processed_sentences:
                    outfile.write(cleaned_line + '\n')
                    logging.debug(f"Written cleaned line to output: '{cleaned_line}'")
                    processed_sentences.add(cleaned_line)  # Track this sentence to prevent duplicates
                elif cleaned_line in processed_sentences:
                    logging.debug(f"Skipped duplicate line: '{cleaned_line}'")
                else:
                    logging.debug("Skipped writing an empty line.")
        
        logging.info(f"Cleaning process completed. Cleaned file saved as {output_file}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    input_file = 'data/datasets/random/random_sentences_1000.txt'
    output_file = 'data/datasets/random/random_sentences_cleaned_replaced_1000.txt'  # Save with a new name

    clean_file(input_file, output_file)

if __name__ == "__main__":
    main()
