import kagglehub
import os
def ensure_dataset_downloaded():
    """
    Ensures the RadioML2016 dataset is downloaded using kagglehub.
    Returns the path to the dataset files.
    """
    print("Checking/Downloading RadioML2016 dataset...")
    path = kagglehub.dataset_download("nolasthitnotomorrow/radioml2016-deepsigcom")
    print(f"Dataset available at: {path}")
    # Symlink to data directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    target_link = os.path.join(data_dir, "radioml2016")
    if os.path.islink(target_link):
        os.remove(target_link)
    elif os.path.exists(target_link):
        print(f"Warning: {target_link} exists and is not a symlink. Keeping it.")
        return path
        
    os.symlink(path, target_link)
    print(f"Symlinked dataset to: {target_link}")
    
    return path
if __name__ == "__main__":
    ensure_dataset_downloaded()