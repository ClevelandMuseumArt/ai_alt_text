import csv
import re
import glob
from pathlib import Path


def combine_csv_files(
    input_pattern, output_file, duplicate_columns, regex_patterns=None
):
    """
    Combine multiple CSV files and remove duplicates using regex matching.

    Args:
        input_pattern: Glob pattern for input files (e.g., 'data/*.csv' or ['file1.csv', 'file2.csv'])
        output_file: Path to output combined CSV file
        duplicate_columns: List of column names to check for duplicates
        regex_patterns: Dict mapping column names to regex patterns for normalization
                       Example: {'email': r'[\s\-\.]', 'phone': r'[\(\)\-\s]'}
    """
    if regex_patterns is None:
        regex_patterns = {}

    # Get list of CSV files
    if isinstance(input_pattern, str):
        csv_files = glob.glob(input_pattern)
    else:
        csv_files = input_pattern

    if not csv_files:
        print("No CSV files found!")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    # Store unique rows
    seen_keys = set()
    all_rows = []
    headers = None
    total_rows = 0

    for file_path in csv_files:
        print(f"Processing: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Get headers from first file
            if headers is None:
                headers = reader.fieldnames
                print(f"Headers: {headers}")

            for row in reader:
                total_rows += 1
                # Create a normalized key for duplicate detection
                key_parts = []
                for col in duplicate_columns:
                    value = row.get(col, "").strip()

                    # Apply regex normalization if pattern provided
                    if col in regex_patterns:
                        value = re.sub(regex_patterns[col], "", value).lower()
                    else:
                        value = value.lower()

                    key_parts.append(value)

                key = tuple(key_parts)

                # Only add if we haven't seen this key before
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_rows.append(row)

    # Write combined data to output file
    print(f"\nWriting {len(all_rows)} unique rows to {output_file}")

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done! Removed {total_rows - len(all_rows)} duplicates")


# Example usage:
if __name__ == "__main__":
    combine_csv_files(
        input_pattern='single_shot_results/*.csv',
        output_file="gemini_3_non_finetuned_single_shot_results.csv",
        duplicate_columns=["image_id"],
    )
