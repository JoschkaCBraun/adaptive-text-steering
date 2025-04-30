import json

def transform_readability_dataset(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract tuples and create the new structure
    tuples = []
    for entry in data.values():
        complex_summary = entry['Complex Summary']
        simple_summary = entry['Simple Summary (Without Context)']
        tuples.append([complex_summary, simple_summary])

    # Create the output dictionary
    output_data = {
        "sample_size": 500,
        "language": "en",
        "dataset": "readability_explanations",
        "tuples": tuples
    }

    # Write the output JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Transformation complete. New dataset saved as {output_file}")

def main():
    input_file = 'data/datasets/readability/readability_synthetic_data/readability_explanations_500_en.json'
    output_file = 'data/datasets/readability/readability_explanations_500_en_transformed.json'

    transform_readability_dataset(input_file, output_file)

if __name__ == "__main__":
    main()