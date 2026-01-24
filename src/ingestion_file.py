"""
Data Ingestion Script
Converts JSONL files to CSV format and saves to input_data folder
"""
import json
import pandas as pd
from pathlib import Path


def load_jsonl(filepath: str) -> list:
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def jsonl_to_csv(jsonl_path: str, csv_path: str):
    """Convert JSONL file to CSV"""
    print(f"Loading {jsonl_path}...")
    data = load_jsonl(jsonl_path)
    
    # Convert to DataFrame
    df_data = []
    for item in data:
        row = {
            'example_id': item['example_id'],
            'input_ids': str(item['input_ids']),
            'attention_mask': str(item['attention_mask'])
        }
        if 'label' in item:
            row['label'] = item['label']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")
    print(f"  Shape: {df.shape}")
    return df


def main():
    """Main ingestion function"""
    print("=" * 60)
    print("Data Ingestion: JSONL to CSV Conversion")
    print("=" * 60)
    
    raw_dir = Path("Raw_data")
    output_dir = Path("input_data")
    output_dir.mkdir(exist_ok=True)
    
    print("\n[1/2] Converting train.jsonl...")
    train_df = jsonl_to_csv(
        jsonl_path=raw_dir / "train.jsonl",
        csv_path=output_dir / "train.csv"
    )
    
    print("\n[2/2] Converting test.jsonl...")
    test_df = jsonl_to_csv(
        jsonl_path=raw_dir / "test.jsonl",
        csv_path=output_dir / "test.csv"
    )
    
    print("\n" + "=" * 60)
    print("✅ Ingestion Complete!")
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")


if __name__ == "__main__":
    main()
