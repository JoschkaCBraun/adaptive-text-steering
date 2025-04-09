import csv
from collections import Counter
import os
import matplotlib.pyplot as plt

def plot_topic_distribution(input_file):
    # Read the CSV and count topic occurrences
    topic_counter = Counter()
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            # Normalize "Music and Sound" category
            topic = "Music and Sound" if row[0].startswith("Music and Sound") else row[0]
            topic_counter[topic] += 1

    # Sort the topics by count in descending order
    sorted_topics = sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)
    topics, counts = zip(*sorted_topics)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(topics, counts)
    plt.title('Distribution of Topics in Sentiment Dataset')
    plt.xlabel('Topics')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    output_file = 'topic_distribution.png'
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

    # Display topic counts
    print("\nTopic Distribution:")
    for topic, count in sorted_topics:
        print(f"{topic}: {count}")

def clean_and_deduplicate_csv(input_file, output_file):
    # Read the CSV file
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Save the header
        rows = list(reader)

    # Normalize "Music and Sound" category
    for row in rows:
        if row[0].startswith("Music and Sound"):
            row[0] = "Music and Sound"

    # Sort rows by topic (alphabetically) and then by the rest of the columns
    sorted_rows = sorted(rows, key=lambda x: (x[0], x[1], x[2], x[3]))

    # Remove duplicates
    unique_rows = []
    seen = set()
    for row in sorted_rows:
        # Create a tuple of (neutral, positive, negative) for checking duplicates
        row_key = tuple(row[1:])
        if row_key not in seen:
            unique_rows.append(row)
            seen.add(row_key)

    # Write the deduplicated data to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(unique_rows)

    # Count occurrences of each topic
    topic_counter = Counter(row[0] for row in unique_rows)

    # Print the new distribution
    print("New Topic Distribution:")
    for topic, count in topic_counter.most_common():
        print(f"{topic}: {count}")

    return topic_counter

def main():
    input_file = 'data/datasets/sentiment/sentiment_synthetic_data/sentiment_words/sentiment_words_1000_en.csv'
    output_file = 'data/datasets/sentiment/sentiment_synthetic_data/sentiment_words/sentiment_words_1000_en_cleaned.csv'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    topic_distribution = clean_and_deduplicate_csv(input_file, output_file)
    print(f"\nCleaning and deduplication complete. CSV file saved as {output_file}")

    plot_topic_distribution(output_file)

if __name__ == "__main__":
    main()