import os
import tempfile
import pandas as pd
import tensorflow as tf
from google.cloud import storage
import numpy as np
from dna_tokenizer import (
    tokenize, 
    detokenize, 
    create_tf_example, 
    verify_tfrecord, 
    STOI, 
    ITOS
)

def test_tokenization():
    """Test basic tokenization and detokenization"""
    print("\nTesting tokenization...")
    
    # Test cases
    test_sequences = [
        "ACGT",  # Basic sequence
        "acgt",  # Lower case
        "NNNN",  # Unknown bases
        "ACGT" * 2048,  # Long sequence (8192 bases)
    ]
    
    for seq in test_sequences:
        tokens = tokenize(seq)
        decoded = detokenize(tokens)
        print(f"\nOriginal: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        print(f"Decoded:  {decoded[:50]}{'...' if len(decoded) > 50 else ''}")
        print(f"Token distribution: {np.unique(tokens, return_counts=True)}")
        assert len(tokens) == len(seq), f"Length mismatch: {len(tokens)} != {len(seq)}"

def test_tf_example_creation():
    """Test creation of TF Examples"""
    print("\nTesting TF Example creation...")
    
    sequence = "ACGT" * 2048  # 8192 bases
    example = create_tf_example(sequence, sequence_length=8192)
    
    # Parse the example back
    feature_description = {
        "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    
    parsed = tf.io.parse_single_example(
        example.SerializeToString(), 
        feature_description
    )
    
    tokens = parsed["x"].numpy()
    segment_ids = parsed["segment_ids"].numpy()
    
    print(f"Sequence length: {len(tokens)}")
    print(f"Token distribution: {np.unique(tokens, return_counts=True)}")
    print(f"Segment IDs distribution: {np.unique(segment_ids, return_counts=True)}")
    
    assert len(tokens) == 8192, f"Wrong sequence length: {len(tokens)}"
    assert len(segment_ids) == 8192, f"Wrong segment IDs length: {len(segment_ids)}"

def test_tfrecord_creation():
    """Test full TFRecord creation and reading"""
    print("\nTesting TFRecord creation...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        test_data = {
            'Sequence': [
                'ACGT' * 2048,  # Perfect length sequence
                'ACGT' * 1024,  # Short sequence (needs padding)
                'N' * 8192,     # All unknown sequence
                'NNNNNACGT' * 819  # Starts with NNNNN (should be skipped)
            ]
        }
        test_df = pd.DataFrame(test_data)
        
        # Save test CSV
        csv_path = os.path.join(tmpdir, 'test.csv')
        test_df.to_csv(csv_path, index=False)
        
        # Create TFRecord
        output_dir = os.path.join(tmpdir, 'tfrecords')
        os.makedirs(output_dir, exist_ok=True)
        
        # Mock GCS bucket read
        def mock_load_csv(*args):
            return test_df
        
        # Process the test data
        from dna_tokenizer import process_genome
        process_genome(csv_path, output_dir, sequence_length=8192, records_per_file=2)
        
        # Verify the output
        tfrecord_files = [f for f in os.listdir(output_dir) if f.endswith('.tfrecord')]
        print(f"\nCreated {len(tfrecord_files)} TFRecord files")
        
        # Check first file
        verify_tfrecord(os.path.join(output_dir, tfrecord_files[0]))

def main():
    print("Starting DNA Tokenizer tests...")
    
    # Run tests
    test_tokenization()
    test_tf_example_creation()
    test_tfrecord_creation()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()