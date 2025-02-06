import os
import re
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import pandas as pd

PADDING = "P"
VOCAB = [PADDING, "U", "A", "C", "G", "T", "N"]  #  padding, unknown, A, C, G, T
VOCAB_SIZE = len(VOCAB)
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}
tokenize = lambda x: [stoi.get(ch, 0) for ch in x]
detokenize = lambda x: "".join([itos.get(i, "U") for i in x])

LAD_CAT = {
    "inter-LAD": 0,
    "LAD": 1,
    "LAD boundary": 2,
}
LAD_CAT_REV = {v:k for k,v in LAD_CAT.items()}

SAD_CAT = {
    "inter-SAD": 0,
    "SAD": 1,
    "SAD boundary": 2,
}
SAD_CAT_REV = {v:k for k,v in SAD_CAT.items()}

CHROM = {
    "chr1": 0,
    "chr2": 1,
    "chr3": 2,
    "chr4": 3,
    "chr5": 4,
    "chr6": 5,
    "chr7": 6,
    "chr8": 7,
    "chr9": 8,
    "chr10": 9,
    "chr11": 10,
    "chr12": 11,
    "chr13": 12,
    "chr14": 13,
    "chr15": 14,
    "chr16": 15,
    "chr17": 16,
    "chr18": 17,
    "chr19": 18,
    "chr20": 19,
    "chr21": 20,
    "chr22": 21,
    "chrX": 22,
    "chrY": 23,
}
CHROM_REV = {v:k for k,v in CHROM.items()}

CELL_TYPE = {
    "intermediate_progenitor": 0,
    "excitatory_neuron": 1,
    "radial_glia": 2,
    "293T": 3,
    "unknown": 4
}
CELL_REV = {v:k for k,v in CELL_TYPE.items()}

def next_multiple(x, n):
    return x + (-x % n)


def process_dfs(ip_df_base, en_df_base, rg_df_base, t293_df_base, sequence_length):
    ip_df_base['cell_type'] = 'intermediate_progenitor'
    en_df_base['cell_type'] = 'excitatory_neuron'
    rg_df_base['cell_type'] = 'radial_glia'
    t293_df_base['cell_type'] = '293T'
    # Check if LMNB1 Cat is 'LAD' (value 1) in all cell types
    lad_conserved = (ip_df_base['LMNB1 Cat'] == 'LAD') & \
                    (en_df_base['LMNB1 Cat'] == 'LAD') & \
                    (rg_df_base['LMNB1 Cat'] == 'LAD') & \
                    (t293_df_base['LMNB1 Cat'] == 'LAD')    

    ip_df_base['lad_conserved'] = lad_conserved
    en_df_base['lad_conserved'] = lad_conserved
    rg_df_base['lad_conserved'] = lad_conserved
    t293_df_base['lad_conserved'] = lad_conserved

    sad_conserved = (ip_df_base['SON Cat'] == 'SAD') & \
                    (en_df_base['SON Cat'] == 'SAD') & \
                    (rg_df_base['SON Cat'] == 'SAD') & \
                    (t293_df_base['SON Cat'] == 'SAD')
    
    ip_df_base['sad_conserved'] = sad_conserved 
    en_df_base['sad_conserved'] = sad_conserved
    rg_df_base['sad_conserved'] = sad_conserved 
    t293_df_base['sad_conserved'] = sad_conserved

    # Shuffle the dataframes
    ip_df = ip_df_base.sample(frac=1, random_state=42).reset_index(drop=True)
    en_df = en_df_base.sample(frac=1, random_state=42).reset_index(drop=True)
    rg_df = rg_df_base.sample(frac=1, random_state=42).reset_index(drop=True)
    t293_df = t293_df_base.sample(frac=1, random_state=42).reset_index(drop=True)

    # Get the trainin sets (first 80% of rows)
    train_ip = ip_df.head(int(len(ip_df) * 0.8))
    train_en = en_df.head(int(len(en_df) * 0.8))
    train_rg = rg_df.head(int(len(rg_df) * 0.8))
    train_t293 = t293_df.head(int(len(t293_df) * 0.8))

    # Get the validation sets (last 20% of rows)
    val_ip = ip_df.tail(int(len(ip_df) * 0.2))
    val_en = en_df.tail(int(len(en_df) * 0.2))
    val_rg = rg_df.tail(int(len(rg_df) * 0.2))
    val_t293 = t293_df.tail(int(len(t293_df) * 0.2))

    # Process individual cell type data
    # Intermediate progenitor cells
    process_rows(train_ip, output_dir=f"gs://minformer_data/go_data_ip_train/tfrecords", bucket=sequence_length)
    process_rows(val_ip, output_dir=f"gs://minformer_data/go_data_ip_val/tfrecords", bucket=sequence_length)

    # Excitatory neurons
    process_rows(train_en, output_dir=f"gs://minformer_data/go_data_en_train/tfrecords", bucket=sequence_length)
    process_rows(val_en, output_dir=f"gs://minformer_data/go_data_en_val/tfrecords", bucket=sequence_length)

    # Radial glia
    process_rows(train_rg, output_dir=f"gs://minformer_data/go_data_rg_train/tfrecords", bucket=sequence_length)
    process_rows(val_rg, output_dir=f"gs://minformer_data/go_data_rg_val/tfrecords", bucket=sequence_length)

    # 293T
    process_rows(train_t293, output_dir=f"gs://minformer_data/go_data_t293_train/tfrecords", bucket=sequence_length)
    process_rows(val_t293, output_dir=f"gs://minformer_data/go_data_t293_val/tfrecords", bucket=sequence_length)

    # Combine into single training and validation sets
    train_combined = pd.concat([train_ip, train_en, train_rg, train_t293], ignore_index=True)
    train_combined = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    val_combined = pd.concat([val_ip, val_en, val_rg, val_t293], ignore_index=True)
    val_combined = val_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Process and save the data
    process_rows(val_combined, output_dir=f"gs://minformer_data/go_data_val/tfrecords/", bucket=sequence_length)
    process_rows(train_combined, output_dir=f"gs://minformer_data/go_data_train/tfrecords/", bucket=sequence_length)


def process_rows(df, output_dir, bucket: int):
    print(f"Processing {output_dir} with {len(df)} rows")
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle the dataframe
    record_count = 0 # Keep track of how many files we've created
    save_together = 128 # Number of sequences to save in each TFRecord file
    save_together_rows = [] # Temporary list to hold rows together before saving

    for i in tqdm(range(0, len(df))): # Loop through rows with progress bar
        row = df.iloc[i] # Accesses a specific row in a pandas df using integer-based location indexing
        if row["Sequence"][0:5] != "NNNNN": # Skip sequences that start with NNNNN (poor quality sequences)
            save_together_rows.append(row)

        if len(save_together_rows) == save_together: # When we have 128 sequences, process them
            # Create a new TFRecord file
            output_file = os.path.join(output_dir, f"record_{record_count}.tfrecord")

            with tf.io.TFRecordWriter(output_file) as writer:
                for row in save_together_rows:
                    # Convert DNA sequence to tokens and add padding
                    tokens = tokenize(row["Sequence"])
                    padding = bucket - len(tokens)
                    segment_ids = np.ones_like(tokens)
                    tokens = np.pad(tokens, (0, padding))
                    segment_ids = np.pad(segment_ids, (0, padding))
                    if "LMNB1 Cat" in row:
                        lad_category = LAD_CAT[row["LMNB1 Cat"]]
                        lad_value = row.get("LMNB1 Signal", 0.0) # Default to 0.0 if missing
                        sad_category = SAD_CAT[row["SON Cat"]]
                        sad_value = row.get("SON Signal", 0.0) # Default to 0.0 if missing
                        chromosome = row["Chrom"]  # Keep chromosome as string
                        # Add conservation information
                        lad_conserved = int(row.get("lad_conserved", False))
                        sad_conserved = int(row.get("sad_conserved", False))
                        cell_type = CELL_TYPE[row.get("cell_type", "unknown")]
                        start_pos = int(row["Start"]) 
                        end_pos = int(row["End"])

                    else:
                        lad_category = 0
                        lad_value = 0
                        sad_category = 0
                        sad_value = 0
                        chromosome = "NA"
                        # Add conservation information
                        lad_conserved = int(row.get("lad_conserved", False))
                        sad_conserved = int(row.get("sad_conserved", False))
                        cell_type = CELL_TYPE[row.get("cell_type", "unknown")]
                        start_pos = 0
                        end_pos = 0

                    save_tfrecord(
                        writer, tokens, segment_ids, lad_category, lad_value,
                        sad_category, sad_value, chromosome, lad_conserved,
                        sad_conserved, cell_type, start_pos, end_pos
                    )

            record_count += 1
            save_together_rows = []
        else:
            pass


def save_tfrecord(writer, tokens, segment_ids, lad_category, lad_value,
                 sad_category, sad_value, chromosome, lad_conserved,
                 sad_conserved, cell_type, start_pos, end_pos):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                # Each feature is stored with a specific data type:

                # Integer lists (for sequences of numbers)
                "x": tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
                "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
                
                # Single integers (wrapped in a list)
                "lad_category": tf.train.Feature(int64_list=tf.train.Int64List(value=[lad_category])),
                
                # Single floating point numbers
                "lad_value": tf.train.Feature(float_list=tf.train.FloatList(value=[lad_value])),
                "sad_category": tf.train.Feature(int64_list=tf.train.Int64List(value=[sad_category])),
                "sad_value": tf.train.Feature(float_list=tf.train.FloatList(value=[sad_value])),
                
                # String (chromosome name) - needs to be encoded to bytes
                "chromosome": tf.train.Feature(bytes_list=tf.train.BytesList(value=[chromosome.encode()])),
                
                # Boolean values (stored as integers)
                "lad_conserved": tf.train.Feature(int64_list=tf.train.Int64List(value=[lad_conserved])),
                "sad_conserved": tf.train.Feature(int64_list=tf.train.Int64List(value=[sad_conserved])),
                "cell_type": tf.train.Feature(int64_list=tf.train.Int64List(value=[cell_type])),
                
                # Genomic coordinates
                "start_pos": tf.train.Feature(int64_list=tf.train.Int64List(value=[start_pos])),
                "end_pos": tf.train.Feature(int64_list=tf.train.Int64List(value=[end_pos]))
            }
        )
    )
    writer.write(example.SerializeToString())


def feature_description() -> Any: # Any indicates this function returns any type
    """Provides scheme for reading TFRecord files and tells TensorFlow how to interpret and parse each field in the TFRecord when loading the data back"""
    return {
        # For sequence data (like DNA sequence)
        "x": tf.io.FixedLenSequenceFeature(
            shape=[], # Empty shape means scalar elements
            dtype=tf.int64, # Each element is a 64-bit integer
            allow_missing=True), # Allow sequences to be missing/variable length
        "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "lad_category": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "lad_value": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "sad_category": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "sad_value": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "chromosome": tf.io.FixedLenFeature([], tf.string),
        "lad_conserved": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "sad_conserved": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "cell_type": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "start_pos": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "end_pos": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }



def load_and_retokenize_tfrecord(file_path: str):
    retokenized_data = [] # Will store DNA sequences
    feature_data = { # Will store all other features
        "segment_ids": [],
        "lad_category": [],
        "lad_value": [],
        "sad_category": [],
        "sad_value": [],
        "chromosome": [],
        "lad_conserved": [],
        "sad_conserved": [],
        "cell_type": [],
        "start_pos": [],
        "end_pos": []
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description())

    dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = dataset.map(_parse_function)

    for parsed_record in parsed_dataset:
        x = parsed_record["x"].numpy()

        original_sequence = detokenize(x)  # Assuming detokenize function is defined elsewhere
        retokenized_data.append(original_sequence)

        for feature in feature_data.keys():
            if feature == "chromosome":
                feature_data[feature].append(parsed_record[feature].numpy().decode())  # Decode bytes to string
            else:
                feature_data[feature].append(parsed_record[feature].numpy())

    return retokenized_data, feature_data


def create_iterator(stage_1: list[str], stage_2: list[str], batch_size: int, shuffle: bool = False):
    """Creates a python iterator to load batches."""

    def _parse_function(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description())
        return parsed_features

    # List all files matching the patterns
    stage_1_files = []
    for pattern in stage_1:
        stage_1_files.extend(tf.io.gfile.glob(pattern))

    # Shuffle the file list
    random.shuffle(stage_1_files)

    print(f"Found {len(stage_1_files)} files for stage 1")

    # Now (for example human genome), we want to have the end of
    # training focused on this.
    stage_2_files = []
    for pattern in stage_2:
        stage_2_files.extend(tf.io.gfile.glob(pattern))

    print(f"Found {len(stage_2_files)} files for stage 2")

    # Shuffle the file list
    random.shuffle(stage_2_files)

    # Combine them.
    files = stage_1_files + stage_2_files
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function, num_parallel_calls=32)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(32)

    for batch in dataset:
        yield {
            "x": batch["x"].numpy().astype(np.int32),
            "segment_ids": batch["segment_ids"].numpy().astype(np.int32),
            "lad_category": batch["lad_category"].numpy().astype(np.int32),
            "lad_value": batch["lad_value"].numpy().astype(np.float32),
            "sad_category": batch["sad_category"].numpy().astype(np.int32),
            "sad_value": batch["sad_value"].numpy().astype(np.float32),
            "chromosome": np.array([batch["chromosome"].numpy()]),
            "lad_conserved": batch["lad_conserved"].numpy().astype(np.int32),
            "sad_conserved": batch["sad_conserved"].numpy().astype(np.int32),
            "cell_type": batch["cell_type"].numpy().astype(np.int32),
            "start_pos": batch["start_pos"].numpy().astype(np.int32),
            "end_pos": batch['end_pos'].numpy().astype(np.int32)
        }

if __name__ == "__main__":
    # Read the CSV files
    ip_df_base = pd.read_csv('gs://minformer_data/eukaryote_pands/8kb_genomic_bins_with_sequences_GW17IPC.csv')
    en_df_base = pd.read_csv('gs://minformer_data/eukaryote_pands/8kb_genomic_bins_with_sequences_GW17eN.csv')
    rg_df_base = pd.read_csv('gs://minformer_data/eukaryote_pands/8kb_genomic_bins_with_sequences_GW17RG.csv')
    t293_df_base = pd.read_csv('gs://minformer_data/eukaryote_pands/8kb_genomic_bins_with_sequences_T293.csv')

    # Print initial lengths
    print(f"Initial lengths:")
    print(f"IP: {len(ip_df_base)}")
    print(f"EN: {len(en_df_base)}")
    print(f"RG: {len(rg_df_base)}")
    print(f"293T: {len(t293_df_base)}")

    # Process the dataframes
    sequence_length = 8192
    process_dfs(ip_df_base, en_df_base, rg_df_base, t293_df_base, sequence_length)