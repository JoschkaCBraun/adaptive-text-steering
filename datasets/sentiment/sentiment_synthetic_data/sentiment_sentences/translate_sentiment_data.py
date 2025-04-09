import csv
import os
from typing import List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def read_csv(file_path: str, n_samples: int = None) -> List[Tuple[str, str]]:
    """Read the CSV file and return a list of tuples, limited to n_samples if specified."""
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [(row['positive'], row['negative']) for row in reader]
    return data[:n_samples] if n_samples is not None else data

def write_csv(file_path: str, data: List[Tuple[str, str]]):
    """Write the translated data to a new CSV file."""
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['positive', 'negative'])
        writer.writerows(data)

def translate_tuples(tuples: List[Tuple[str, str]], source_lang: str, target_lang: str) -> List[Tuple[str, str]]:
    """Translate a list of tuples from a source language to a target language with a progress bar."""
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    translation_pipeline = pipeline(task='translation', model=model, tokenizer=tokenizer, device='mps')
    
    translated_tuples = []
    for pos, neg in tqdm(tuples, desc=f"Translating to {target_lang}", unit="pair"):
        translated_pos = translation_pipeline(pos, max_length=512)[0]['translation_text']
        translated_neg = translation_pipeline(neg, max_length=512)[0]['translation_text']
        translated_tuples.append((translated_pos, translated_neg))
    
    return translated_tuples

def main():
    # Set up file paths
    input_file = 'data/datasets/sentiment/sentiment_500_en.csv'
    output_dir = 'data/datasets/sentiment/'
    
    # Set the number of samples to translate
    n_samples = 500  # Change this to the desired number of samples
    
    # Define target languages
    target_languages = ['es', 'it', 'ru']
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the English CSV file
    print(f"Reading the first {n_samples} samples from {input_file}")
    english_data = read_csv(input_file, n_samples)
    
    # Translate to each target language
    for lang in target_languages:
        print(f"\nTranslating to {lang}...")
        translated_data = translate_tuples(english_data, 'en', lang)
        
        # Write the translated data to a new CSV file
        output_file = os.path.join(output_dir, f"sentiment_500_{lang}.csv")
        print(f"Writing translated data to {output_file}")
        write_csv(output_file, translated_data)
        
        print(f"Translation to {lang} complete.")

    print("\nAll translations complete.")

if __name__ == "__main__":
    main()