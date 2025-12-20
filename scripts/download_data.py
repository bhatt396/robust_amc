"""Script to help download and prepare the dataset"""
import os
import requests
import tarfile
import pickle
import numpy as np

def download_rml_dataset():
    """
    Download RML2016.10a dataset from DeepSig.
    Note: This requires manual download as the dataset is behind a form.
    """
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("RML2016.10a Dataset Download Instructions")
    print("=" * 60)
    print("\n1. Go to: https://www.deepsig.ai/datasets")
    print("2. Click on 'RML2016.10a' dataset")
    print("3. Fill out the form and submit")
    print("4. Download the dataset (will be sent to your email)")
    print("5. Extract the downloaded file")
    print("6. Look for 'RML2016.10a_dict.pkl' file")
    print(f"\n7. Place the file in: {os.path.abspath(data_dir)}/")
    print("\nNote: The file should be named: RML2016.10a_dict.pkl")
    print("\nAfter placing the file, run: python experiments/train_baseline.py")
    print("=" * 60)
    
    # Check if file already exists
    expected_path = os.path.join(data_dir, "RML2016.10a_dict.pkl")
    if os.path.exists(expected_path):
        print(f"\n✓ Dataset found at: {expected_path}")
        print(f"  File size: {os.path.getsize(expected_path) / (1024*1024):.2f} MB")
        
        # Try to load and verify
        try:
            with open(expected_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            print(f"  ✓ Successfully loaded dataset")
            print(f"  ✓ Contains {len(data)} modulation types")
            print(f"  ✓ Example modulation: {list(data.keys())[0]}")
            return True
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            return False
    else:
        print(f"\n✗ Dataset not found at: {expected_path}")
        print("  Please follow the instructions above to download and place the file.")
        return False

def create_sample_dataset():
    """Create a small sample dataset for testing if full dataset is not available"""
    print("\nCreating a small sample dataset for testing...")
    
    sample_data = {
        'BPSK': {
            -20: [np.random.randn(2, 128) + 1j*np.random.randn(2, 128) for _ in range(10)],
            0: [np.random.randn(2, 128) + 1j*np.random.randn(2, 128) for _ in range(10)],
            20: [np.random.randn(2, 128) + 1j*np.random.randn(2, 128) for _ in range(10)]
        },
        'QPSK': {
            -20: [np.random.randn(2, 128) + 1j*np.random.randn(2, 128) for _ in range(10)],
            0: [np.random.randn(2, 128) + 1j*np.random.randn(2, 128) for _ in range(10)],
            20: [np.random.randn(2, 128) + 1j*np.random.randn(2, 128) for _ in range(10)]
        }
    }
    
    sample_path = "data/sample_dataset.pkl"
    with open(sample_path, 'wb') as f:
        pickle.dump(sample_data, f)
    
    print(f"✓ Created sample dataset at: {sample_path}")
    print("  Note: This is synthetic data for testing only.")
    print("  For real experiments, download the full RML2016.10a dataset.")
    
    return sample_path

if __name__ == "__main__":
    # Try to download/check for real dataset
    if not download_rml_dataset():
        # If real dataset not found, create sample
        create = input("\nCreate a sample dataset for testing? (y/n): ")
        if create.lower() == 'y':
            create_sample_dataset()