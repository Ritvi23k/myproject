"""
Data preparation script for Tiny Shakespeare dataset.
Uses character-level tokenization for fast CPU training.
"""
import os
import numpy as np

# --- Configuration ---
INPUT_FILE = "input.txt"
OUTPUT_DIR = os.path.join("data", "shakespeare")
TRAIN_RATIO = 0.9

def main():
    # Read the raw text
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  Total characters: {len(text):,}")

    # Build character-level vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"  Unique characters: {vocab_size}")
    print(f"  Vocabulary: {''.join(chars)}")

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}

    # Save vocabulary for generation script
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUTPUT_DIR, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("".join(chars))
    print(f"  Saved vocab to {meta_path}")

    # Encode the text
    tokens = np.array([stoi[ch] for ch in text], dtype=np.uint16)
    print(f"  Total tokens: {len(tokens):,}")

    # Split into train and val
    split_idx = int(len(tokens) * TRAIN_RATIO)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens:   {len(val_tokens):,}")

    # Save to disk
    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path = os.path.join(OUTPUT_DIR, "val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"\nSaved:")
    print(f"  {train_path} ({os.path.getsize(train_path):,} bytes)")
    print(f"  {val_path} ({os.path.getsize(val_path):,} bytes)")
    print(f"\nVocab size: {vocab_size}")
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
