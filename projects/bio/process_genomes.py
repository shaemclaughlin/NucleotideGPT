import argparse
import os
import pandas as pd
from google.cloud import storage
from tqdm import tqdm
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dna_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDNADataset:
    def __init__(self, sequence_length: int = 8192):
        self.sequence_length = sequence_length
        self.bases = ["A", "C", "G", "T"]
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.bases)}
        self.stoi["<unk>"] = 0
        self.itos = {i + 1: ch for i, ch in enumerate(self.bases)}
        self.itos[0] = "<unk>"

    def tokenize(self, sequence):
        return np.array([self.stoi.get(base, 0) for base in sequence], dtype=np.int32)

    def process_and_save_tfrecord(self, row, writer):
        """Process a single row and save as TFRecord."""
        sequence = str(row['Sequence'])  # Note the capital S in Sequence
        
        if len(sequence) != self.sequence_length:
            return False, f"Sequence length mismatch: {len(sequence)} != {self.sequence_length}"

        tokens = self.tokenize(sequence)
        
        # Create feature dictionary with all fields
        feature = {
            'x': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
            'segment_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=np.ones_like(tokens))),
            'chrom': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(row['Chrom']).encode()])),
            'start': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['Start']])),
            'end': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['End']])),
            'contains_n': tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if row['Contains_N'] else 0]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        return True, None

def process_file(input_file, bucket_name, dataset_name, test_mode=False):
    """Process a single file and return statistics."""
    # Read input file
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(input_file)
    content = blob.download_as_string()
    df = pd.read_csv(pd.io.common.BytesIO(content), skiprows=1)  # Skip the duplicated header
    
    logger.info(f"Loaded file {input_file}")
    logger.info(f"Columns found: {df.columns.tolist()}")
    logger.info(f"Total sequences in file: {len(df)}")
    
    # In test mode, only process first few rows
    if test_mode:
        logger.info("Running in test mode - processing only first 3 rows")
        df = df.head(3)
    
    # Setup output directory
    output_dir = f"gs://{bucket_name}/diverse_genomes_tf_records/{dataset_name}/tfrecords"
    
    # Initialize counters
    stats = {
        'total_sequences': len(df),
        'successful_conversions': 0,
        'failed_conversions': 0,
        'errors': []
    }
    
    # Create DNA dataset processor
    processor = EnhancedDNADataset()
    
    # Process in batches
    batch_size = 1000 if not test_mode else 1
    for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {dataset_name}"):
        batch_df = df.iloc[i:i+batch_size]
        output_file = f"{output_dir}/batch_{i//batch_size}.tfrecord"
        
        with tf.io.TFRecordWriter(output_file) as writer:
            for _, row in batch_df.iterrows():
                success, error = processor.process_and_save_tfrecord(row, writer)
                if success:
                    stats['successful_conversions'] += 1
                else:
                    stats['failed_conversions'] += 1
                    stats['errors'].append(error)
                    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Process genomic datasets")
    parser.add_argument("--bucket-name", type=str, default="minformer_data")
    parser.add_argument("--test", action="store_true", help="Run in test mode with one small file")
    args = parser.parse_args()

    # Get list of files
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket_name)
    blobs = list(bucket.list_blobs(prefix="eukaryote_pands/"))
    
    # Process each file and collect statistics
    results = {}
    
    # In test mode, just process the first CSV file found
    if args.test:
        test_file = next(blob.name for blob in blobs if blob.name.endswith('.csv'))
        dataset_name = os.path.splitext(os.path.basename(test_file))[0]
        logger.info(f"Testing with file: {test_file}")
        
        try:
            stats = process_file(test_file, args.bucket_name, dataset_name, test_mode=True)
            results[dataset_name] = stats
            
            # Log results
            logger.info(f"\nResults for {dataset_name}:")
            logger.info(f"Total sequences processed: {stats['total_sequences']}")
            logger.info(f"Successful conversions: {stats['successful_conversions']}")
            logger.info(f"Failed conversions: {stats['failed_conversions']}")
            if stats['errors']:
                logger.info("Sample of errors:")
                for error in stats['errors'][:5]:
                    logger.info(f"  - {error}")
                    
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")
            results[dataset_name] = {'error': str(e)}

    # Save final report
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'processing_report_{report_time}.txt', 'w') as f:
        for dataset, stats in results.items():
            f.write(f"\n{'-'*50}\n")
            f.write(f"Dataset: {dataset}\n")
            for key, value in stats.items():
                if key != 'errors':
                    f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()