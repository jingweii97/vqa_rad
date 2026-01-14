import argparse
import torch
import os
import json
from torch.utils.data import DataLoader

from src.common.config import DEVICE, EMBED_DIM, LSTM_HIDDEN, ATTENTION_DIM, IMAGE_DIR, TEST_BATCH_SIZE
from src.san.dataset import load_data, VQARadDataset, get_transforms
from src.san.model import SAN_RAD
from src.san.train import train_san
from src.san.evaluate import evaluate
from src.common.utils import set_seed, build_answer_vocab
from src.san.visualize import plot_training_curves, plot_test_metrics
# Import qualitative analysis script
from src.script.analyze_qualitative import analyze_qualitative_results

def main():
    parser = argparse.ArgumentParser(description="VQA-RAD SAN Baseline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model on test set")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for evaluation")
    parser.add_argument("--use_biobert", action="store_true", help="Enable BioBERT semantic similarity for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load shared data and vocabs
    print("Loading data...")
    train_loader, val_loader, test_data, vocab, ans_vocab = load_data()

    # Initialize Model
    model = SAN_RAD(
        vocab_size=len(vocab),
        ans_vocab_size=len(ans_vocab),
        embed_dim=EMBED_DIM,
        lstm_hidden=LSTM_HIDDEN,
        attention_dim=ATTENTION_DIM
    ).to(DEVICE)

    # 1. Training Mode
    if args.train:
        print("\n=== TRAINING MODE ===")
        history = train_san(model, train_loader, val_loader, ans_vocab)
        
        # Save training curves
        print("\nGenerating training curves...")
        plot_training_curves(history, save_dir='results')

    # 2. Evaluation Mode
    if args.eval:
        print("\n=== EVALUATION MODE ===")
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            # Default to the one saved during training if not specified
            from src.common.config import MODEL_SAVE_PATH
            if os.path.exists(MODEL_SAVE_PATH):
                checkpoint_path = MODEL_SAVE_PATH
            else:
                print("Error: No checkpoint specified and no default model found. Use --checkpoint.")
                return

        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            return

        # Prepare Test Loader
        _, test_transform = get_transforms()
        test_ds = VQARadDataset(test_data, IMAGE_DIR, vocab, ans_vocab, test_transform)
        test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # Evaluate and get stats
        stats = evaluate(model, test_loader, ans_vocab, model_type='san', return_stats=True, use_biobert=args.use_biobert)
        
        # Create results directory for SAN-RAD
        results_dir = 'results/san'
        os.makedirs(results_dir, exist_ok=True)
        
        # Export metrics to JSON
        print(f"\nExporting evaluation metrics to {results_dir}/metrics.json...")
        metrics_export = {
            "model": "SAN-RAD",
            "closed": {
                "total": stats["closed"]["total"],
                "correct": stats["closed"]["correct"],
                "accuracy": stats["closed"]["correct"] / max(stats["closed"]["total"], 1)
            },
            "open": {
                "total": stats["open"]["total"],
                "correct": stats["open"]["correct"],
                "accuracy": stats["open"]["correct"] / max(stats["open"]["total"], 1),
                "bleu": stats["open"]["bleu_sum"] / max(stats["open"]["total"] - stats["open"]["unk_label"], 1)
            },
            "overall": {
                "total": stats["closed"]["total"] + stats["open"]["total"],
                "correct": stats["closed"]["correct"] + stats["open"]["correct"],
                "accuracy": (stats["closed"]["correct"] + stats["open"]["correct"]) / 
                           max(stats["closed"]["total"] + stats["open"]["total"], 1)
            }
        }
        
        # Add BioBERT metrics if computed
        if args.use_biobert and "biobert_sum" in stats["open"]:
            valid_open = stats["open"]["total"] - stats["open"]["unk_label"]
            metrics_export["open"]["biobert_avg"] = stats["open"]["biobert_sum"] / max(valid_open, 1)
            metrics_export["open"]["biobert_strict"] = stats["open"]["biobert_strict"]
            metrics_export["open"]["biobert_soft"] = stats["open"]["biobert_soft"]
        
        with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_export, f, indent=2)
        print(f"✅ Metrics exported to {results_dir}/metrics.json")
        
        # Save test metrics visualization
        if stats:
            print(f"\nGenerating test metrics visualization...")
            plot_test_metrics(stats, save_dir=results_dir)
            
        # Plot training curves if history exists
        history_path = os.path.join(os.path.dirname(checkpoint_path), 'training_history.json')
        if os.path.exists(history_path):
            print(f"Found training history at {history_path}. Plotting curves...")
            with open(history_path, 'r') as f:
                history = json.load(f)
            plot_training_curves(history, save_dir=results_dir)
            
        # Run Qualitative Analysis
        print("\nRunning Qualitative Analysis (Attention Maps)...")
        analyze_qualitative_results(
            model=model,
            test_data=test_data,
            vocab=vocab,
            ans_vocab=ans_vocab,
            num_examples=6,
            save_dir=os.path.join(results_dir, 'qualitative')
        )
        
        print(f"\n{'='*60}")
        print(f"✅ All results saved to: {results_dir}/")
        print(f"   - metrics.json (evaluation metrics)")
        print(f"   - training_curves.png")
        print(f"   - test_accuracy_breakdown.png")
        print(f"   - qualitative/ (attention maps)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
