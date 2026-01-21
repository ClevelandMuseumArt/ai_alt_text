import pandas as pd
import json
import sys
from pathlib import Path


def analyze_csv(file_path):
    """Analyze a single CSV file and extract statistics."""
    df = pd.read_csv(file_path)

    # Basic statistics
    total_images = len(df)

    # Inner product statistics
    highest_inner_product = df["inner_product"].max()
    lowest_inner_product = df["inner_product"].min()
    avg_inner_product = df["inner_product"].mean()
    median_inner_product = df["inner_product"].median()

    # Cosine similarity statistics
    highest_cosine_similarity = df["cosine_similarity"].max()
    lowest_cosine_similarity = df["cosine_similarity"].min()
    avg_cosine_similarity = df["cosine_similarity"].mean()
    median_cosine_similarity = df["cosine_similarity"].median()

    # Top and bottom 10 for inner product (including caption)
    top_10_inner_product = df.nlargest(10, "inner_product")[
        ["image_id", "caption", "inner_product"]
    ].to_dict("records")
    bottom_10_inner_product = df.nsmallest(10, "inner_product")[
        ["image_id", "caption", "inner_product"]
    ].to_dict("records")

    # Top and bottom 10 for cosine similarity (including caption)
    top_10_cosine_similarity = df.nlargest(10, "cosine_similarity")[
        ["image_id", "caption", "cosine_similarity"]
    ].to_dict("records")
    bottom_10_cosine_similarity = df.nsmallest(10, "cosine_similarity")[
        ["image_id", "caption", "cosine_similarity"]
    ].to_dict("records")

    return {
        "file_name": Path(file_path).name,
        "total_images": total_images,
        "inner_product": {
            "highest": highest_inner_product,
            "lowest": lowest_inner_product,
            "average": avg_inner_product,
            "median": median_inner_product,
            "top_10": top_10_inner_product,
            "bottom_10": bottom_10_inner_product,
        },
        "cosine_similarity": {
            "highest": highest_cosine_similarity,
            "lowest": lowest_cosine_similarity,
            "average": avg_cosine_similarity,
            "median": median_cosine_similarity,
            "top_10": top_10_cosine_similarity,
            "bottom_10": bottom_10_cosine_similarity,
        },
    }


def main():
    # Get input files from command line arguments
    if len(sys.argv) < 3:
        print(
            "Usage: python script.py <output_file.json> <input_file1.csv> [input_file2.csv ...]"
        )
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    results = []

    # Process each input file
    for file_path in input_files:
        try:
            print(f"Processing {file_path}...")
            result = analyze_csv(file_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    # Write results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis complete! Results written to {output_file}")
    print(f"Processed {len(results)} file(s)")


if __name__ == "__main__":
    main()
