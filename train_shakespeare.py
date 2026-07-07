"""
train_shakespeare.py — Train a small GPT on Tiny Shakespeare from scratch.
Uses character-level tokenization for fast CPU training.
Adapted from Karpathy's train_gpt2.py architecture.
"""
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# =============================================================================
# Model definition (same architecture as GPT-2, just smaller config)
# =============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256     # context length
    vocab_size: int = 65      # character-level (~65 unique chars in Shakespeare)
    n_layer: int = 6          # 6 transformer layers
    n_head: int = 6           # 6 attention heads
    n_embd: int = 384         # 384 embedding dimension (~10M params total)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # type: ignore

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # type: ignore
        tok_emb = self.transformer.wte(idx) # type: ignore
        x = tok_emb + pos_emb
        for block in self.transformer.h: # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# =============================================================================
# Character-level tokenizer
# =============================================================================

class CharTokenizer:
    def __init__(self, meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            chars = f.read()
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens):
        return "".join([self.itos[t] for t in tokens])

# =============================================================================
# Data loader — simple single-file version
# =============================================================================

def load_tokens(filename):
    tokens = np.fromfile(filename, dtype=np.uint16)
    tokens = tokens.astype(np.int32)
    return torch.tensor(tokens, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        data_dir = os.path.join("data", "shakespeare")
        filename = os.path.join(data_dir, f"{split}.bin")
        self.tokens = load_tokens(filename)
        print(f"loaded {len(self.tokens):,} tokens for {split}")
        self.current_position = 0

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# =============================================================================
# Training configuration
# =============================================================================

if __name__ == "__main__":
    # --- Device detection ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Load vocab
    meta_path = os.path.join("data", "shakespeare", "meta.txt")
    enc = CharTokenizer(meta_path)
    print(f"vocab size: {enc.vocab_size}")

    # --- Hyperparameters ---
    B = 4          # micro batch size (benchmarked: ~670ms/step on CPU)
    T = 256        # sequence length (= block_size)
    total_batch_size = B * T  # 1024 tokens per step
    grad_accum_steps = 1
    print(f"total batch size: {total_batch_size:,} tokens")

    # Create data loaders
    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader = DataLoaderLite(B=B, T=T, split="val")

    # Create model with actual vocab size
    config = GPTConfig(vocab_size=enc.vocab_size)
    model = GPT(config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {total_params:,}")

    # --- Optimizer and LR schedule ---
    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100
    max_steps = 5000

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, device_type=device_type)

    # --- Logging setup ---
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, "w") as f:
        pass

    # ==========================================================================
    # Training loop
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Starting training for {max_steps} steps (~{total_params/1e6:.1f}M params, char-level)")
    print(f"{'='*60}\n")

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # --- Validation loss every 250 steps ---
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 10
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            print(f"step {step:5d} | val loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # --- Generate a sample every 500 steps ---
        if (step > 0 and step % 500 == 0) or last_step:
            model.eval()
            max_length = 200
            tokens = enc.encode("First Citizen:\n")
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)
            with torch.no_grad():
                while tokens.size(1) < max_length:
                    idx_cond = tokens if tokens.size(1) <= config.block_size else tokens[:, -config.block_size:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / 0.8
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, min(50, probs.size(-1)), dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    tokens = torch.cat((tokens, xcol), dim=1)
            decoded = enc.decode(tokens[0].tolist())
            print(f"  --- sample ---")
            print(decoded)
            print(f"  --- end sample ---")

        # --- Training step ---
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = total_batch_size / dt
        if step % 50 == 0 or last_step:
            print(f"step {step:5d} | train loss: {loss_accum.item():.4f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.0f}ms | tok/sec: {tokens_per_sec:.0f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

    # ==========================================================================
    # Save final checkpoint
    # ==========================================================================
    checkpoint_path = "ckpt.pt"
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'step': max_steps,
        'val_loss': val_loss_accum.item(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    print("Training complete!")
