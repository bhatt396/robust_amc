"""Main entry point for the SNR-Adaptive AMC project"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='SNR-Adaptive AMC using Mixture of Experts')
    
    parser.add_argument('--task', type=str, default='train',
                       choices=['train', 'evaluate', 'demo', 'download'],
                       help='Task to perform')
    parser.add_argument('--model', type=str, default='moe',
                       choices=['baseline', 'moe'],
                       help='Model to train/evaluate')
    parser.add_argument('--data_path', type=str, default='data/RML2016.10a_dict.pkl',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SNR-Adaptive AMC using Mixture of Experts")
    print("=" * 60)
    
    if args.task == 'download':
        print("\nDownloading dataset...")
        from scripts.download_data import download_rml_dataset
        download_rml_dataset()
        
    elif args.task == 'train':
        if args.model == 'baseline':
            print(f"\nTraining baseline model for {args.epochs} epochs...")
            from experiments.train_baseline import train_baseline_model
            train_baseline_model(args.data_path, args.epochs, args.batch_size)
        else:
            print(f"\nTraining MoE model for {args.epochs} epochs...")
            from experiments.train_moe import train_moe_model
            train_moe_model()
            
    elif args.task == 'evaluate':
        print("\nEvaluating models...")
        from experiments.evaluate import main as evaluate_main
        evaluate_main()
        
    elif args.task == 'demo':
        print("\nLaunching demo...")
        # You would implement a Gradio/Streamlit demo here
        print("Demo functionality coming soon!")
        # from app.demo import launch_demo
        # launch_demo()
    
    print("\n" + "=" * 60)
    print("Task completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()