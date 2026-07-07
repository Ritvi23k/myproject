"""
generate.py — Load a trained GPT checkpoint and generate Shakespeare-like text.
Uses character-level tokenization.
"""
import torch
from train_shakespeare import GPT, GPTConfig, CharTokenizer
import os

def main():
    import sys
    # --- Configuration ---
    checkpoint_path = "ckpt.pt"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "O Romeo, Romeo! wherefore art thou Romeo?"
    max_new_tokens = 250
    temperature = 0.8
    top_k = 50

    # --- Device detection ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # --- Load tokenizer ---
    meta_path = os.path.join("data", "shakespeare", "meta.txt")
    enc = CharTokenizer(meta_path)
    print(f"vocab size: {enc.vocab_size}")

    # --- Load checkpoint ---
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"Model config: {config}")
    print(f"Trained for {checkpoint['step']} steps, val loss: {checkpoint['val_loss']:.4f}")

    # --- Build model and load weights ---
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # --- Tokenize prompt ---
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)

    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...\n")
    print("=" * 60)

    # --- Generate ---
    torch.manual_seed(42)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = tokens if tokens.size(1) <= config.block_size else tokens[:, -config.block_size:]
            # Forward pass
            logits, _ = model(idx_cond)
            # Take logits at the last position and apply temperature
            logits = logits[:, -1, :] / temperature
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)

    # --- Decode and print ---
    generated_text = enc.decode(tokens[0].tolist())
    print(generated_text)
    print("=" * 60)
    print(f"\nGeneration complete! ({tokens.size(1)} total tokens)")

if __name__ == "__main__":
    main()
