"""
Complete pipeline script to run BYOL self-supervised learning.
This script orchestrates the entire process from dataset preparation to evaluation.
"""


import argparse
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ Error!")
        print("Error:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Complete BYOL Pipeline')
    
    # Pipeline control
    parser.add_argument('--skip_preparation', action='store_true',
                       help='Skip dataset preparation step')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip BYOL training step')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation step')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Backbone architecture')
    
    # Paths
    parser.add_argument('--output_dir', type=str, default='outputs/byol',
                       help='Output directory for training')
    parser.add_argument('--eval_output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation')
    
    args = parser.parse_args()
    
    print("🚀 Starting BYOL Self-Supervised Learning Pipeline")
    print(f"Backbone: {args.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Step 1: Dataset Preparation
    if not args.skip_preparation:
        success = run_command(
            "python prepare_dataset.py",
            "Dataset Preparation"
        )
        if not success:
            print("❌ Dataset preparation failed. Exiting.")
            return
    
    # Step 2: BYOL Training
    if not args.skip_training:
        train_command = f"""
        python train_byol.py \
            --train_data_dir data/ssl_dataset/train_unified \
            --val_data_dir data/ssl_dataset/val_unified \
            --output_dir {args.output_dir} \
            --backbone {args.backbone} \
            --epochs {args.epochs} \
            --batch_size {args.batch_size} \
            --learning_rate 3e-4 \
            --weight_decay 1e-6 \
            --num_workers 4 \
            --save_every 10
        """
        
        success = run_command(train_command, "BYOL Self-Supervised Training")
        if not success:
            print("❌ Training failed. Exiting.")
            return
    
    # Step 3: Evaluation
    if not args.skip_evaluation:
        # Find the best model checkpoint
        output_path = Path(args.output_dir)
        best_model_path = output_path / "best_model.pth"
        
        if not best_model_path.exists():
            print(f"❌ Best model not found at {best_model_path}")
            print("Available files:")
            for file in output_path.glob("*.pth"):
                print(f"  - {file}")
            return
        
        eval_command = f"""
        python evaluate_byol.py \
            --model_path {best_model_path} \
            --backbone {args.backbone} \
            --batch_size {args.batch_size} \
            --ft_epochs 50 \
            --ft_lr 1e-3 \
            --output_dir {args.eval_output_dir}
        """
        
        success = run_command(eval_command, "Model Evaluation")
        if not success:
            print("❌ Evaluation failed.")
            return
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"Training outputs: {args.output_dir}")
    print(f"Evaluation outputs: {args.eval_output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print("✓ Dataset preparation completed")
    print("✓ BYOL self-supervised training completed")
    print("✓ Model evaluation completed")
    print("\nNext steps:")
    print("1. Check the training history plots in the output directory")
    print("2. Review the evaluation results and confusion matrices")
    print("3. Use the pretrained backbone for your downstream tasks")


if __name__ == '__main__':
    main()
