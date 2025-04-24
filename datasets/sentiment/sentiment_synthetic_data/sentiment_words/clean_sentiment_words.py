import csv
from collections import Counter, defaultdict
import os
import matplotlib.pyplot as plt
import sys # Import sys for error handling exit

def plot_topic_distribution(input_file):
    """
    Reads a cleaned CSV file, counts topic occurrences, generates
    a bar plot, and saves it in the same directory as the input file.
    """
    print(f"Attempting to plot distribution from: {input_file}")
    topic_counter = Counter()
    header = []
    try:
        # Ensure file exists before trying to open
        if not os.path.exists(input_file):
             print(f"Error: Input file for plotting not found at {input_file}")
             return

        with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            try:
                header = next(reader)  # Read the header row
                # print(f"Plotting based on header: {header}")
            except StopIteration:
                print(f"Error: File '{input_file}' is empty. Cannot generate plot.")
                return # Cannot plot an empty file

            for i, row in enumerate(reader):
                if not row: # Skip empty rows if any
                    # print(f"Warning: Skipping empty row {i+2} in {input_file}")
                    continue
                if len(row) == 0: # Check specifically for rows with zero columns
                    # print(f"Warning: Skipping row {i+2} with zero columns in {input_file}")
                    continue
                # Topic is the first column (index 0)
                try:
                    topic = row[0]
                    topic_counter[topic] += 1
                except IndexError:
                    print(f"Warning: Skipping row {i+2}. Cannot access topic at index 0 in row: {row}")
                    continue

    except FileNotFoundError: # Should be caught by os.path.exists, but belt-and-suspenders
        print(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading {input_file} for plotting: {e}")
        return

    if not topic_counter:
        print("No topics found in the data rows. Cannot generate plot.")
        return

    # Sort the topics by count in descending order
    try:
        sorted_topics = sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)
        if not sorted_topics: # Should not happen if topic_counter is not empty, but check
             print("Error: Could not sort topics.")
             return
        topics, counts = zip(*sorted_topics)
    except Exception as e:
        print(f"Error preparing data for plotting: {e}")
        return


    # --- Create the plot ---
    try:
        plt.figure(figsize=(14, 8)) # Slightly wider figure for potentially many topics
        plt.bar(topics, counts)
        plt.title(f'Distribution of Topics\n(Source: {os.path.basename(input_file)})', fontsize=14)
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Number of Occurrences', fontsize=12)
        # Adjust rotation and font size for better readability
        plt.xticks(rotation=80, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # --- Save the plot ---
        # Construct output plot path in the same directory as the input CSV
        input_dir = os.path.dirname(input_file)
        base_filename = os.path.splitext(os.path.basename(input_file))[0] # e.g., "sentiment_words_2800_en_cleaned"
        # Ensure unique plot filename if script is run multiple times on same input? Not strictly needed now.
        output_file_plot = os.path.join(input_dir or '.', f"{base_filename}_topic_distribution.png") # Use current dir if input_dir is empty

        # Ensure the plot directory exists (it must if input file exists, but safer)
        if input_dir:
            os.makedirs(input_dir, exist_ok=True)

        plt.savefig(output_file_plot)
        print(f"Plot saved successfully as {output_file_plot}")
        plt.close() # Close the plot figure to free memory

    except Exception as e:
        print(f"An error occurred during plot creation or saving: {e}")
        # Ensure plot is closed even if saving fails
        plt.close()
        return # Indicate plotting failed

    # Display topic counts from the cleaned file
    print("\nFinal Topic Distribution (from cleaned file):")
    for topic, count in sorted_topics:
        print(f"{topic}: {count}")

def clean_sort_deduplicate_csv(input_file, output_file):
    """
    Reads the specified input CSV, cleans it by removing duplicates based on the
    (positive, negative) pair using topic counts for arbitration,
    sorts it alphabetically, and saves the result to output_file.
    Relies on actual files existing.
    """
    rows = []
    header = []
    initial_topic_counts = Counter()
    pair_occurrences = defaultdict(list) # Stores rows grouped by (pos, neg) pair

    # --- Step 1: Read data, count initial topics, group by (pos, neg) pair ---
    print(f"Attempting to read input file: {input_file}")
    try:
        # Explicitly check if input file exists
        if not os.path.isfile(input_file):
             print(f"Error: Input file not found at {input_file}")
             return None # Indicate critical failure

        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            try:
                 header = next(reader)  # Save the header
            except StopIteration:
                 print(f"Error: Input file {input_file} is empty.")
                 # Decide if an empty output file with header should be created
                 try:
                     # Ensure output directory exists before writing empty header file
                     output_dir = os.path.dirname(output_file)
                     if output_dir:
                         os.makedirs(output_dir, exist_ok=True)
                     with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                         # Write empty header if needed? Or just leave output empty?
                         # Let's not write header to signal input was truly empty.
                         pass # Create empty file
                     print(f"Created empty output file: {output_file}")
                 except Exception as e_write:
                     print(f"Error trying to create empty output file {output_file}: {e_write}")
                 return Counter() # Return empty counter for empty input

            if len(header) < 4:
                 print(f"Error: CSV file header must have at least 4 columns (topic, neutral, positive, negative). Found: {header}")
                 return None # Indicate critical failure

            # Process rows
            for i, row in enumerate(reader):
                line_num = i + 2 # Account for header row
                if not row:
                    # print(f"Warning: Skipping empty row {line_num}") # Reduce noise maybe
                    continue
                if len(row) < 4:
                    print(f"Warning: Skipping row {line_num} due to insufficient columns (< 4): {row}")
                    continue

                # Extract fields - ensure they are treated as strings initially
                topic = str(row[0]).strip() # Add strip() for robustness
                neutral = str(row[1]).strip()
                positive = str(row[2]).strip()
                negative = str(row[3]).strip()
                # Consider if empty strings after stripping are valid? Assume yes for now.
                full_row = [topic, neutral, positive, negative] # Store standardized row

                # Count initial topics
                initial_topic_counts[topic] += 1

                # Group rows by (positive, negative) pair
                pair_key = (positive, negative)
                pair_occurrences[pair_key].append(full_row)
                rows.append(full_row) # Keep track of all valid rows read

    except FileNotFoundError: # Should be caught by isfile check, but redundant check
        print(f"Error: Input file not found at {input_file}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_file}: {e}")
        return None # Indicate critical failure

    if not rows:
        print("No valid data rows found in the input file after header.")
        # Create empty file with header (if header was read)
        try:
            if header:
                 output_dir = os.path.dirname(output_file)
                 if output_dir:
                     os.makedirs(output_dir, exist_ok=True)
                 with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                 print(f"Output file created with only header: {output_file}")
            else:
                 # If even header failed, maybe create totally empty file?
                 with open(output_file, 'w') as f: pass # Create empty file
                 print(f"Created empty output file (no header read): {output_file}")

        except Exception as e_write:
             print(f"Error trying to create header-only/empty output file {output_file}: {e_write}")
        return Counter() # Return empty counter

    print(f"\nInitial Topic Counts (Total {len(rows)} valid rows read):")
    for topic, count in initial_topic_counts.most_common():
        print(f"- {topic}: {count}")

    # --- Step 2: Apply Deduplication Logic ---
    rows_to_keep = []
    processed_pairs = set() # Keep track if needed, but logic below handles pairs directly

    print("\nProcessing duplicates based on (positive, negative) pairs...")
    for pair, associated_rows in pair_occurrences.items():
        if len(associated_rows) == 1:
            # Unique pair, keep the row
            rows_to_keep.append(associated_rows[0])
        else:
            # Duplicate pair found, decide which row to keep
            # print(f"  Duplicate pair found: {pair}. Occurrences: {len(associated_rows)}") # Reduce noise
            min_topic_count = float('inf')
            candidates_for_keeping = []

            # Find the minimum initial topic count among the rows for this pair
            for row in associated_rows:
                topic = row[0]
                count = initial_topic_counts.get(topic, 0) # Use initial counts
                # print(f"    - Candidate row: {row}, Topic: '{topic}', Initial Count: {count}")
                if count < min_topic_count:
                    min_topic_count = count
                    candidates_for_keeping = [row] # New minimum found
                    # print(f"      -> New minimum count {min_topic_count}. Candidate list reset.")
                elif count == min_topic_count:
                    candidates_for_keeping.append(row) # Add to list of candidates with same min count
                    # print(f"      -> Same minimum count {min_topic_count}. Added to candidates.")

            # Decide based on candidates
            if len(candidates_for_keeping) == 1:
                row_to_keep = candidates_for_keeping[0]
                # print(f"    -> Keeping row (only one at min count {min_topic_count}): {row_to_keep}")
            else:
                # Tie-breaking: Sort candidates alphabetically and keep the first one
                # print(f"    -> Tie-breaking needed for {len(candidates_for_keeping)} candidates with count {min_topic_count}.")
                candidates_for_keeping.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
                row_to_keep = candidates_for_keeping[0]
                # print(f"    -> Keeping row (first alphabetically after tie-break): {row_to_keep}")

            rows_to_keep.append(row_to_keep)

    print(f"Processing duplicates complete. Kept {len(rows_to_keep)} rows.")

    # --- Step 3: Final Sort ---
    # Sort the rows selected for keeping alphabetically
    rows_to_keep.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    print("Final selected rows sorted alphabetically.")

    # --- Step 4: Write Output ---
    print(f"Attempting to write cleaned data to: {output_file}")
    try:
        # Ensure the output directory exists before writing
        output_dir = os.path.dirname(output_file)
        if output_dir: # Only create if it's not the current directory
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write the header
            writer.writerows(rows_to_keep)
        print(f"Cleaned data successfully written to {output_file}")
    except Exception as e:
        print(f"FATAL ERROR: An error occurred writing the final data to {output_file}: {e}")
        return None # Indicate critical failure

    # --- Step 5: Calculate and return final topic distribution ---
    final_topic_counter = Counter(row[0] for row in rows_to_keep)

    # Final counts will be printed by the plotting function reading the clean file.
    # Returning the counter might still be useful for other purposes if needed.
    return final_topic_counter

def main():
    # Updated file paths - using 'datasets/' directly at the top level
    input_file = 'datasets/sentiment/sentiment_synthetic_data/sentiment_words/sentiment_words_2800_en_original.csv'
    output_file = 'datasets/sentiment/sentiment_synthetic_data/sentiment_words/sentiment_words_2800_en_cleaned.csv'

    print("--- Starting Sentiment Data Cleaning Script ---")
    print(f"Input file:  {os.path.abspath(input_file)}")
    print(f"Output file: {os.path.abspath(output_file)}")

    # --- Run the cleaning process ---
    final_topic_distribution = clean_sort_deduplicate_csv(input_file, output_file)

    # --- Post-cleaning actions ---
    if final_topic_distribution is None:
        print("\n--- Script finished with critical errors during cleaning. ---")
        sys.exit(1) # Exit with error code
    elif not final_topic_distribution and os.path.exists(output_file):
        # Check if file exists and counter is empty (e.g. header only or empty file)
        file_size = os.path.getsize(output_file)
        if file_size <= len(','.join(header)) + 2 : # Approx check if only header exists
             print(f"\n--- Cleaning process finished. Output file '{output_file}' contains no data rows (or only header). Skipping plot. ---")
        else:
             # Should not happen if counter is empty, but catch potential edge case
              print(f"\n--- Cleaning process finished. Output file '{output_file}' created, but no topics found. Skipping plot. ---")

    elif final_topic_distribution:
        print(f"\n--- Cleaning process finished successfully. {len(final_topic_distribution)} topics found in cleaned data. ---")
        print("\nGenerating topic distribution plot...")
        # Plotting function reads the definitive cleaned file
        plot_topic_distribution(output_file)
        print("\n--- Script finished successfully. ---")
    else:
         # Catch case where counter is empty but file might not exist (though clean func should handle this)
         print("\n--- Cleaning process finished, but no topics found and output file might be missing. Check logs. ---")


if __name__ == "__main__":
    main()