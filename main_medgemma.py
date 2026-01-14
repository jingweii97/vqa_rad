"""
MedGemma Main Entry Point

Run zero-shot evaluation of MedGemma-4B on VQA-RAD.
"""
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="MedGemma Zero-Shot VQA-RAD Evaluation")
    parser.add_argument("--hf_token", type=str, default=None, 
                       help="HuggingFace token for model access")
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it",
                       help="MedGemma model ID")
    parser.add_argument("--no_biobert", action="store_true",
                       help="Skip BioBERT semantic evaluation")
    parser.add_argument("--output", type=str, default="results/medgemma_results.json",
                       help="Output path for results JSON")
    args = parser.parse_args()
    
    # Set HF token if provided
    if args.hf_token:
        import os
        os.environ["HF_TOKEN"] = args.hf_token
    
    # Import after setting token
    from src.medgemma.evaluate import load_medgemma_model, evaluate_medgemma
    
    print("="*60)
    print("MEDGEMMA ZERO-SHOT VQA-RAD EVALUATION")
    print("="*60)
    
    # Load model
    model, processor = load_medgemma_model(model_id=args.model_id)
    
    # Run evaluation
    metrics = evaluate_medgemma(
        model, processor,
        use_biobert=not args.no_biobert
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        # Remove per_sample for summary file (can be large)
        summary = {k: v for k, v in metrics.items() if k != "per_sample"}
        json.dump(summary, f, indent=2)
    print(f"\n📁 Results saved to {args.output}")
    
    # Also save per-sample predictions
    per_sample_path = args.output.replace(".json", "_detailed.json")
    with open(per_sample_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📁 Detailed results saved to {per_sample_path}")


if __name__ == "__main__":
    main()
