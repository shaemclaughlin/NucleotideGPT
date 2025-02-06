from google.cloud import storage
from collections import defaultdict

def summarize_datasets():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('minformer_data')
    blobs = bucket.list_blobs(prefix='diverse_genomes_tf/')
    
    # Group files by dataset
    datasets = defaultdict(lambda: {'count': 0, 'total_size': 0})
    
    for blob in blobs:
        # Get dataset name from path (assuming format diverse_genomes_tf/dataset_name/...)
        parts = blob.name.split('/')
        if len(parts) > 2:
            dataset = parts[1]
            datasets[dataset]['count'] += 1
            datasets[dataset]['total_size'] += blob.size
    
    # Print summary
    print("\nDataset Summary:")
    print("-" * 60)
    print(f"{'Dataset Name':<30} {'File Count':<15} {'Total Size'}")
    print("-" * 60)
    for dataset, info in sorted(datasets.items()):
        size_gb = info['total_size'] / (1024**3)  # Convert to GB
        print(f"{dataset:<30} {info['count']:<15} {size_gb:.2f} GB")

if __name__ == "__main__":
    summarize_datasets()