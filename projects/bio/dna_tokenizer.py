import os
import time
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta

# Define vocabulary constants
BASES = ["A", "C", "G", "T"]
VOCAB = ["<pad>", "<unk>"] + BASES  # Add padding and unknown tokens
VOCAB_SIZE = len(VOCAB)
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for i, ch in enumerate(VOCAB)}

def tokenize(sequence):
    """Convert DNA sequence to token integers."""
    return [STOI.get(base, STOI["<unk>"]) for base in sequence.upper()]

def detokenize(tokens):
    """Convert token integers back to DNA sequence."""
    return "".join([ITOS.get(token, "<unk>") for token in tokens])

def load_csv_from_gcp(bucket_name, file_name):
    """Load CSV file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_string()
    df = pd.read_csv(pd.io.common.BytesIO(content))
    return df

def create_tf_example(sequence, sequence_length):
    """Create a TF Example from a DNA sequence."""
    # Tokenize sequence
    tokens = tokenize(sequence)
    
    # Pad or truncate to desired length
    if len(tokens) < sequence_length:
        padding = sequence_length - len(tokens)
        tokens = tokens + [STOI["<pad>"]] * padding
    else:
        tokens = tokens[:sequence_length]
    
    # Create segment IDs (1s for actual sequence, 0s for padding)
    segment_ids = [1] * len(tokens)
    
    # Create TF Example
    feature = {
        "x": tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def format_time(seconds):
    """Convert seconds to a human-readable format."""
    return str(timedelta(seconds=int(seconds)))

def process_genome(input_file, output_dir, sequence_length=8192, records_per_file=128):
    """Process genome data and create TF records."""
    start_time = time.time()
    print(f"Processing {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_csv_from_gcp("minformer_data", input_file)
    
    # Initialize counters
    record_count = 0
    file_count = 0
    current_records = []
    
    # Process each sequence
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sequence = row["Sequence"]
        
        # Skip sequences that start with NNNNN
        if sequence.startswith("NNNNN"):
            continue
            
        # Create TF Example
        tf_example = create_tf_example(sequence, sequence_length)
        current_records.append(tf_example)
        
        # Write when we have enough records
        if len(current_records) >= records_per_file:
            output_file = os.path.join(output_dir, f"record_{file_count}.tfrecord")
            with tf.io.TFRecordWriter(output_file) as writer:
                for record in current_records:
                    writer.write(record.SerializeToString())
            
            file_count += 1
            record_count += len(current_records)
            current_records = []
    
    # Write any remaining records
    if current_records:
        output_file = os.path.join(output_dir, f"record_{file_count}.tfrecord")
        with tf.io.TFRecordWriter(output_file) as writer:
            for record in current_records:
                writer.write(record.SerializeToString())
        record_count += len(current_records)
    
    processing_time = time.time() - start_time
    processing_speed = record_count / processing_time if processing_time > 0 else 0
    
    print(f"\nProcessing stats for {input_file}:")
    print(f"Time taken: {format_time(processing_time)}")
    print(f"Records created: {record_count}")
    print(f"Processing speed: {processing_speed:.2f} records/second")
    
    return record_count, processing_time

def verify_tfrecord(file_path):
    """Verify TF Record file by reading and detokenizing a sequence."""
    feature_description = {
        "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    
    dataset = tf.data.TFRecordDataset(file_path)
    
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)
    
    parsed_dataset = dataset.map(_parse_function)
    
    # Get first example
    for parsed_record in parsed_dataset.take(1):
        tokens = parsed_record["x"].numpy()
        sequence = detokenize(tokens)
        print(f"\nVerification of {file_path}:")
        print(f"First 100 bases: {sequence[:100]}")
        print(f"Token distribution: {np.unique(tokens, return_counts=True)}")
        break

def main():
    # List of genomes to process
    genomes = [
        'Bradyrhizobium_japonicum_8192bp_bins_no_N',
        'Burkholderia_pseudomallei_8192bp_bins_no_N',
        'Caenorhabditis_elegans_8192bp_bins_no_N',  
        'Combined_viruses_8192bp_bins_no_N',
        'Mixed_viruses_set2_8192bp_bins_no_N',
        'Monodelphis_domestica_8192bp_bins_no_N',
        'Ornithorhynchus_anatinus_8192bp_bins_no_N',
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
    
    sequence_length = 8192
    base_input_path = "eukaryote_pands"
    base_output_path = "gs://minformer_data/diverse_genomes_tf"
    
    # Initialize tracking variables
    total_records = 0
    total_time = 0
    genome_stats = []
    
    overall_start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for genome in genomes:
        input_file = f"{base_input_path}/{genome}.csv"
        output_dir = f"{base_output_path}/{genome}/tfrecords"
        
        print(f"\nProcessing {genome}")
        records, processing_time = process_genome(input_file, output_dir, sequence_length)
        total_records += records
        total_time += processing_time
        
        # Store stats for this genome
        genome_stats.append({
            'genome': genome,
            'records': records,
            'time': processing_time,
            'speed': records / processing_time if processing_time > 0 else 0
        })
        
        # Verify first TF Record
        first_record = f"{output_dir}/record_0.tfrecord"
        verify_tfrecord(first_record)
    
    # Calculate and display final statistics
    overall_time = time.time() - overall_start_time
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*50)
    print("FINAL PROCESSING STATISTICS")
    print("="*50)
    print(f"Start time: {start_timestamp}")
    print(f"End time: {end_timestamp}")
    print(f"Total processing time: {format_time(overall_time)}")
    print(f"Pure processing time: {format_time(total_time)}")
    print(f"Overhead time: {format_time(overall_time - total_time)}")
    print(f"Total records created: {total_records}")
    print(f"Overall speed: {total_records / overall_time:.2f} records/second")
    
    print("\nPer-genome Statistics:")
    print("-"*50)
    for stat in genome_stats:
        print(f"\nGenome: {stat['genome']}")
        print(f"Records: {stat['records']}")
        print(f"Time: {format_time(stat['time'])}")
        print(f"Speed: {stat['speed']:.2f} records/second")

if __name__ == "__main__":
    main()