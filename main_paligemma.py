"""
Main entry point for PaliGemma2 training and evaluation.

Usage:
    Train: python main_paligemma.py --train --hf_token YOUR_TOKEN
    Eval:  python main_paligemma.py --eval --checkpoint model/best_paligemma_adapter
"""
import argparse
import os
import json
from src.common.config import DEVICE, MODEL_DIR
from src.paligemma.dataset import load_paligemma_data
from src.paligemma.train import load_paligemma_model, train_paligemma
from src.paligemma.evaluate import evaluate_paligemma
from src.common.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="VQA-RAD PaliGemma2 Baseline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model on test set")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to adapter checkpoint for evaluation")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for model access")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 1. Training Mode
    if args.train:
        print("\n=== PALIGEMMA TRAINING MODE ===")
        
        if args.hf_token is None:
            print("ERROR: --hf_token is required for training")
            print("Get your token from: https://huggingface.co/settings/tokens")
            return
        
        # Load model
        print("\nLoading PaliGemma2 model...")
        model, processor = load_paligemma_model(hf_token=args.hf_token)
        
        # Load data
        print("\nLoading VQA-RAD data...")
        train_data, val_data, test_data, collate_fn = load_paligemma_data(processor)
        
        # Train
        trainer = train_paligemma(
            model=model,
            train_data=train_data,
            val_data=val_data,
            collate_fn=collate_fn,
            resume_from_checkpoint=args.resume
        )
        
        print("\n✅ Training complete!")
        print(f"Best adapter saved to: {os.path.join(MODEL_DIR, 'best_paligemma_adapter')}")
    
    # 2. Evaluation Mode
    if args.eval:
        print("\n=== PALIGEMMA EVALUATION MODE ===")
        
        # Determine checkpoint path
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            default_path = os.path.join(MODEL_DIR, "best_paligemma_adapter")
            if os.path.exists(default_path):
                checkpoint_path = default_path
                print(f"Using default checkpoint: {checkpoint_path}")
            else:
                print("ERROR: No checkpoint specified and no default found")
                print("Use --checkpoint PATH to specify adapter location")
                return
        
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            return
        
        # Load model with adapter
        print(f"\nLoading model from {checkpoint_path}...")
        from peft import PeftModel
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
        import torch
        
        # Read PEFT config to get base model
        peft_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(peft_config_path):
            with open(peft_config_path) as f:
                peft_cfg = json.load(f)
                base_model_id = peft_cfg.get("base_model_name_or_path")
        else:
            from src.common.config import VLM_MODEL_ID
            base_model_id = VLM_MODEL_ID
        
        # Load base model in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=args.hf_token
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        processor = PaliGemmaProcessor.from_pretrained(base_model_id, token=args.hf_token)
        
        print("Model loaded successfully!\n")
        
        # Load test data
        print("Loading test data...")
        _, _, test_data, _ = load_paligemma_data(processor)
        
        # Evaluate
        stats = evaluate_paligemma(model, processor, test_data, device=DEVICE)
        
        # Save results
        results_path = os.path.join(MODEL_DIR, "paligemma_evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n Results saved to: {results_path}")


if __name__ == "__main__":
    main()
