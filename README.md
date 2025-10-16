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

## Getting Started
### Prerequisites
- Python 3.8+
- Google Cloud account with TPU access
- Google Cloud Storage bucket for data and checkpoints

### Installation
**1. Clone the repository**
```
git clone https://github.com/shaemclaughlin/NucleotideGPT.git
cd NucleotideGPT
```

**2. Install dependencies**
This project uses Poetry for dependency management.
Install Poetry first:
```
curl -sSL https://install.python-poetry.org | python3 -
```
Then install project dependencies:
```
poetry install
```
For TPU support, ensure JAX is properly configured:
```
poetry add jax[tpu] --source jax-releases
```

**3. Set up Google Cloud credentials**
```
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Data Preparation
The model requires softmasked genomic sequences where lowercase letters indicate repetitive elements (from RepeatMasker) and uppercase letters indicate unique sequences.

**Preparing softmasked genome data**
1. Obtain softmasked genome: Download human genome (GRCh38) with RepeatMasker annotations
2. Create CSV file: Format as columns: Chrom, Start, End, Sequence (8192bp chunks)
3. Upload to GCS:
```
gsutil cp hg38_softmasked_8192bp_bins.csv gs://YOUR_BUCKET/human_softmasked/
```

**Creating TFRecords**
We provide three tokenization strategies:
1. Single-nucleotide tokenization (best performance)
```
python create_softmasked_tfrecords.py
```
2. Overlapping 6-mer tokenization
```
python create_tfrecords_softmasked_human_6mer_overlap.py --bucket-name YOUR_BUCKET
```
3. Non-overlapping 6-mer tokenization
```
python create_tfrecords_softmasked_human_6mer_nonoverlap.py --bucket-name YOUR_BUCKET
```
### Training
**Pretraining with RE weighting**
Our code facilitates downweighting repetitive elements during training. To reproduce experiments:
1. Edit RE weight in modelling/model.py:
- 0.0 = complete exclusion of repetitive elements
- 0.1 = 90% downweighting
- 0.5 = 50% downweighting (best performance)
- 1.0 = no downweighting
2. Start training:
For nucleotide-level model:
```
python train.py --batch_size 16 --checkpoint_dir gs://YOUR_BUCKET/checkpoints/nucleotide_0.5 --total_steps 20000 --log_every 50
```
For 6-mer tokenized model:
```
python train_6mer.py --batch_size 16 --checkpoint_dir gs://YOUR_BUCKET/checkpoints/6mer_overlap_0.5 --vocab_size 4098 --total_steps 20000
```

**Training on TPU**
1. Create TPU VM:
```
gcloud compute tpus tpu-vm create YOUR_TPU_NAME --zone=us-central1-a --accelerator-type=v3-8 --version=tpu-vm-tf-2.11.0
```
2. SSH to TPU VM:
```
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone=us-central1-a
```
3. Run training (inside TPU VM):
```
cd NucleotideGPT/projects/bio
python train.py --checkpoint_dir gs://YOUR_BUCKET/checkpoints/
```

**Monitoring Training**
Training progress is logged to:
- WandB
- TensorBoard

**Resume from Checkpoint**
To continue training from a previous run:
```
python train.py --resume_from_checkpoint --checkpoint_dir gs://YOUR_BUCKET/checkpoints/nucleotide_0.5
```

### Finetuning
After pretraining, finetune on genomic classification tasks from the Genomic Benchmarks dataset.
Example usage:
```
python finetune.py \
  --dataset_name human_ensembl_regulatory \
  --pretrained_checkpoint_dir gs://YOUR_BUCKET/checkpoints/nucleotide_0.5 \
  --finetuned_checkpoint_dir gs://YOUR_BUCKET/finetuned/human_ensembl_regulatory \
  --num_classes 3 \
  --num_epochs 1 \
  --batch_size 16 \
  --max_lr 2e-5
```
This script will:
- Load data from Google Cloud Storage
- Initialize a classification head on top of the pretrained model
- Train for the specified number of epochs
- Evaluate on the test set with MCC, F1, and accuracy metrics
- Save the finetuned checkpoint

### Model Comparisons
To compare Nucleotide GPT with other published genomic language models, we provide a Google Colab notebook for benchmarking baseline models on the same Genomic Benchmarks tasks.

Running comparisons:
1. Open benchmarking.ipynb in Google Colab
2. Set runtime to GPU
3. Update the BUCKET_NAME variable with your GCS bucket
4. Run all cells
   
This notebook benchmarks:
- DNABERT (6-mer tokenization)
- HyenaDNA (character-level, long-context)
- Nucleotide Transformer (500M parameter model)
  
These models require PyTorch and GPU, unlike Nucleotide GPT which uses JAX/TPU.

**Contact Details:**
If you have any questions or issues with this code, please contact me at shae.m.mclaughlin@gmail.com.
