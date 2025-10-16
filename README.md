# Nucleotide GPT: A Decoder-Only Transformer for Genomic Sequences

## Overview
Nucleotide GPT is a decoder-only transformer model specifically designed for genomic sequence modeling. This implementation adapts the minimal transformer architecture from <u>Minformer</u> by Sholto Douglas for genomic data processing and generation. 

**Key Features:**
- Efficient transformer architecture optimized for genomic sequences
- Single-nucleotide tokenization preserving biological resolution
- Weighted loss for repetitive elements during pretraining
- TPU-compatible implementation using JAX
- Sparse autoencoder (SAE) for interpretability analysis
- Support for both pre-training and fine-tuning on genomic tasks

## Model Architecture 
Nucleotide GPT employs a LLaMA-style decoder-only transformer with the following specifications:
- 12 transformer layers with model dimension of 2048
- Single-nucleotide tokenization for maximum biological resolution
- Rotary Positional Embeddings (RoPE) for position encoding
- RMSNorm before attention and feed-forward blocks
- Flash Attention for efficient training
- 500M parameters
