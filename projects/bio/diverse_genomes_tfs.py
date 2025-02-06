"""
Example command: python3 diverse_genomes_tfs.py --dataset Bradyrhizobium_japonicum_8192bp_bins_no_N --bucket-name minformer_data
"""

import argparse # For parsing command line arguments
import io # For handling byte streams
import os # For file/path operations
import urllib.request # For downloading files from URLs
from tqdm import tqdm # For tracking
import numpy as np
import tensorflow as tf

import data # Handles basic DNA data processing
import data_shae # Handles custom data processing
import pandas as pd # For data manipulation
from google.cloud import storage # For GCS operations

DIVERSE_GENOMES = [
    'Bradyrhizobium_japonicum_8192bp_bins_no_N',
    'Burkholderia_pseudomallei_8192bp_bins_no_N',
    'Caenorhabditis_elegans_8192bp_bins_no_N',
    'Combined_viruses_8192bp_bins_no_N',
    'Ornithorhynchus_anatinus_8192bp_bins_no_N',
    'Monodelphis_domestica_8192bp_bins_no_N',
    'Pseudomonas_fluorescens_8192bp_bins_no_N',
    'Rhodococcus_jostii_8192bp_bins_no_N',
    'bacteroides_genome_8192bp_bins_no_N',
    'borrelia_genome_8192bp_bins_no_N',
    'candida_genome_8192bp_bins_no_N',
    'chlamydia_genome_8192bp_bins_no_N',
    'drosophila_genome_8192bp_bins_no_N',
    'ecoli_genome_8192bp_bins_no_N', 
    'haloferax_genome_8192bp_bins_no_N',
    'human_genome_8192bp_bins_no_N',
    'macaque_genome_8192bp_bins_no_N',
    'methanogen_genome_8192bp_bins_no_N',
    'mouse_genome_8192bp_bins_no_N',
    'myxococcus_genome_8192bp_bins_no_N',
    'pombe_genome_8192bp_bins_no_N',
    'pseudomonas_genome_8192bp_bins_no_N',
    'pylori_genome_8192bp_bins_no_N',
    'saccharomyces_cerevisiae_8192bp_bins_no_N',
    'streptomyces_genome_8192bp_bins_no_N',
    'subtilis_genome_8192bp_bins_no_N',
    'sulfolobus_genome_8192bp_bins_no_N',
    'tb_genome_8192bp_bins_no_N',
    'thermus_genome_8192bp_bins_no_N',
    'zebrafish_genome_8192bp_bins_no_N'
]

def parse_args():
    # Sets up command-line argument parsing
    parser = argparse.ArgumentParser(description="Download and process DNA data")

    # The --dataset argument lets you choose with dataset to process
    # Can use any of the genomes in the DIVERSE_GENOMES list
    parser.add_argument(
        "--dataset", # The flag used in command line
        type=str, # Type of input expected
        choices=DIVERSE_GENOMES, # Valid options
        default=DIVERSE_GENOMES[0], # What to use if no dataset specified
        help="Type of dataset to download and process", # Help message
    )

    # --use-gcs flag determines if output goes to Google Cloud Storage
    parser.add_argument("--use-gcs", action="store_true", help="Use Google Cloud Storage")
    
    # --bucket-name specifies which GCS bucket to use
    parser.add_argument("--bucket-name", type=str, help="GCS bucket name")
    
    # --sequence-length sets the DNA sequence length (default 8192)
    parser.add_argument("--sequence-length", type=int, default=8192, help="Training seqlen")
    
    return parser.parse_args()

def load_csv_from_gcp_bucket(bucket_name, file_name):
    # Initialize the Google Cloud Storage client
    # Creates a client object that can interact with Google Cloud Storage
    storage_client = storage.Client()

    # Accesses your specific storage bucket ('minformer_data')
    bucket = storage_client.get_bucket(bucket_name)

    # Locates specific CSV file 'blob'
    blob = bucket.blob(file_name)

    # Download the contents of the blob as a string of bytes
    data = blob.download_as_string()

    # Convert the string to a file-like object
    # data.decode("utf-8") converts bytes to a text string
    # io.StringIO() creates a file-like object that pandas can read
    data_file = io.StringIO(data.decode("utf-8"))

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(data_file)

    return df

PADDING = "P"
VOCAB = [PADDING, "A", "C", "G", "T", "N"]
VOCAB_SIZE = len(VOCAB)
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}
tokenize = lambda x: [stoi.get(ch, 0) for ch in x]   
detokenize = lambda x: "".join([itos.get(i, "U") for i in x])

def save_tfrecord(writer, tokens, segment_ids):
    """
    Creates and writes a TFRecord example containing DNA sequence data.

    Converts tokenized DNA sequence and its segment IDs into TensorFlow's binary TFRecord format.
    This format is optimized for training deep learning models with TensorFlow.

    Args:
        writer: TFRecordWriter object for writing to file
        tokens (numpy.array): Array of integers representing the DNA sequence
        segment_ids (numpy.array): Array of 1s and 0s marking real sequence vs padding
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                # DNA sequence as integers
                "x": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=tokens)
                ),

                # Marks which positions are real DNA (1) vs padding (0)
                "segment_ids": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=segment_ids)
                )
            }
        )
    )
    # Convert to binary format and write to file
    writer.write(example.SerializeToString())


def process_rows(df, output_dir, bucket: int):
    """
    Processes DNA sequences from a DataFrame into TFRecord files for model training.

    Takes DNA sequences from a pandas DataFrame and converts them into TensorFlow's binary TFRecord format. 
    Sequences are tokenized, padded to a fixed length, and saved in batches of 128 sequences per file. 
    Sequences with more than 100 Ns are skipped.

    Args:
        df (pandas.DataFrame): DataFrame containing DNA sequences in a 'Sequence' column
        output_dir (str): Directory path where TFRecord files will be saved
        bucket (int): Length to pad/truncate sequences to (typically 8192)
    
    Files created:
        Creates numbered TFRecord files.
        Each file contains 128 sequences in tokenized and padded format.
    """
    print(f"Processing {output_dir} with {len(df)} rows")
    record_count = 0 # Keep track of how many files we've created
    save_together = 128 # Number of sequences to save in each TFRecord file
    save_together_rows = [] # Temporary list to hold rows before saving

    for i in tqdm(range(0, len(df))): # Loop through rows with progress bar
        row = df.iloc[i] # Get one row from DataFrame
        sequence = row["Sequence"] # Get the DNA sequence from that row

        # Skip if row has more than 100 Ns
        if sequence.count('N') <= 100:
            save_together_rows.append(row)
        
        if len(save_together_rows) == save_together: # When we have 128 sequencdes, process them
            output_file = os.path.join(output_dir, f"record_{record_count}.tfrecord")

            # Open TFRecord file for writing
            with tf.io.TFRecordWriter(output_file) as writer:
                # Process each of the 128 sequences
                for row in save_together_rows:
                    # Convert ACGT sequence to numbers
                    tokens = tokenize(row["Sequence"])
                    # Calculate how much padding needed to reach bucket size
                    padding = bucket - len(tokens)
                    # Create array of 1s same length as tokens
                    segment_ids = np.ones_like(tokens)
                    # Pad tokens array with zeros at end
                    tokens = np.pad(tokens, (0, padding))
                    # Pad segment_ids array with zeros at end
                    segment_ids = np.pad(segment_ids, (0, padding))

                    # Write this sequence to TFRecord file
                    save_tfrecord(
                        writer, tokens, segment_ids
                    )
            
            record_count += 1 # Increment file counter
            save_together_rows = [] # Clear temporary list for next batch

def main():
    args = parse_args()

    # Set up output directory
    output_dir = f"gs://{args.bucket_name}/diverse_genomes_tf_v2/{args.dataset}/tfrecords/"
    print(f"Creating packed records at {output_dir}...")

    bucket_name = "minformer_data"

    file_name = f"eukaryote_pands/{args.dataset}.csv"
    print("Loading CSV...")

    df = load_csv_from_gcp_bucket(bucket_name, file_name)

    process_rows(df, output_dir=output_dir, bucket=args.sequence_length)

    print("Packed records created successfully.")

if __name__ == "__main__":
    main()