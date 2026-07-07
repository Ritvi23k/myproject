# 🧠 GPT-2 (124M) Reproduction: Tiny Shakespeare

A from-scratch, local implementation of the OpenAI GPT-2 (124M) language model, trained on the Tiny Shakespeare dataset. 

## 👨‍💻 About the Author & The Journey

Hi, I'm **Ritvik Suri**, currently studying at **IIT (BHU) Varanasi**. 

This project represents the culmination of a long and rewarding learning journey. I've been following Andrej Karpathy's Youtube channel for quite some time. Instead of rushing to the final implementation, I intentionally took the long road: starting with the absolute basics, breaking down Transformer architectures, understanding self-attention mechanisms, and wrapping my head around the core math before finally building this full-scale implementation. 

## 🚀 Project Overview

This repository contains a fully functional training and generation pipeline based on the original GPT-2 architecture. It has been specifically optimized to run on local hardware constraints (like a standard laptop environment) while successfully learning the linguistic structures of William Shakespeare.

### Key Features
* **Architecture:** Standard GPT-2 124M .
* **Dataset:** Tiny Shakespeare, tokenized using the official `gpt2` BPE encoding via `tiktoken`.
* **Hardware Optimizations:** * Implemented gradient accumulation to simulate larger batch sizes without triggering Out-Of-Memory (OOM) errors.
  * Reduced `block_size` and utilized mixed-precision training (`bfloat16`/`float16`) for faster local epochs.
* **Custom Data Loader:** Adapted the pipeline to read from locally compiled binary files (`.bin`) rather than massive web-scraped datasets.

## 🛠️ Implementation Steps

1. **Data Preparation (`data_prepare.py`):** Downloads the Shakespeare text, tokenizes it, and splits it into 90/10 train/validation sets saved as binary streams.
2. **Model Training (`train_gpt2.py`):** The core training loop utilizing AdamW optimization, cosine learning rate decay, and weight decay.
3. **Text Generation (`generate.py`):** Loads the trained weights (`ckpt.pt`) and uses `torch.multinomial` sampling to generate original text based on learned probabilities.

## 💻 How to Run

### 1. Install Dependencies
Ensure you have the required packages installed:
```bash
pip install torch numpy transformers tiktoken datasets