"""
VLM Evaluation Module for PaliGemma2

Evaluates trained PaliGemma2 model on VQA-RAD test set.
"""
import os
import torch
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.common.config import IMAGE_DIR
from src.paligemma.dataset import normalize_answer
from src.common.semantic_eval import BioBERTEvaluator


def evaluate_paligemma(model, processor, test_data, device="cuda", use_biobert=False):
    """
    Evaluate VLM model on test set.
    
    Args:
        model: Trained PaliGemma model
        processor: PaliGemma processor
        test_data: List of test samples
        device: Device to run inference on
        use_biobert: Whether to compute BioBERT similarity for open-ended
        
    Returns:
        stats: Dictionary with evaluation statistics
    """
    model.eval()
    
    # Separate by question type
    closed_test = [r for r in test_data if r["answer_type"].lower() == "closed"]
    open_test = [r for r in test_data if r["answer_type"].lower() == "open"]
    
    print(f"\n{'='*60}")
    print("PALIGEMMA EVALUATION")
    print(f"{'='*60}")
    print(f"Closed questions: {len(closed_test)}")
    print(f"Open questions: {len(open_test)}")
    print(f"Total: {len(test_data)}")
    print(f"BioBERT enabled: {use_biobert}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Initialize BioBERT evaluator if needed
    biobert_eval = None
    if use_biobert:
        biobert_eval = BioBERTEvaluator(device=device)
    
    # Evaluate closed-ended questions
    print("🔍 Evaluating CLOSED-ENDED questions...\n")
    closed_stats = evaluate_closed(model, processor, closed_test, device)
    
    # Evaluate open-ended questions
    print("\n🔍 Evaluating OPEN-ENDED questions...\n")
    open_stats = evaluate_open(model, processor, open_test, device, biobert_eval=biobert_eval)
    
    # Overall statistics
    total_correct = closed_stats["correct"] + open_stats["correct"]
    total_questions = closed_stats["total"] + open_stats["total"]
    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_questions})")
    print(f"Closed Accuracy: {closed_stats['accuracy']:.4f}")
    print(f"Open Accuracy: {open_stats['accuracy']:.4f}")
    if "bleu" in open_stats:
        print(f"Open BLEU Score: {open_stats['bleu']:.4f}")
    print(f"{'='*60}\n")
    
    # Generate visualizations
    try:
        visualize_results(closed_stats['results'], open_stats['results'])
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    
    return {
        "closed": closed_stats,
        "open": open_stats,
        "overall": {
            "correct": total_correct,
            "total": total_questions,
            "accuracy": overall_acc
        }
    }


def evaluate_closed(model, processor, test_data, device):
    """Evaluate closed-ended (yes/no) questions"""
    total = 0
    correct = 0
    
    # Store examples for visualization
    results_list = []
    
    with torch.no_grad():
        for r in tqdm(test_data, desc="Closed-Ended"):
            # Load image
            img_path = os.path.join(IMAGE_DIR, r["image_name"])
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            
            # Prepare input - Add <image> token explicitly to fix warning
            # Prepare input
            question = r["question"]
            # FIX: Training script did not use <image> token manually
            prompt = question
            gt_answer = normalize_answer(r["answer"])
            
            # Generate prediction
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False)
            decoded = processor.decode(outputs[0], skip_special_tokens=True).lower().strip()
            
            # Extract prediction - remove question if echoed
            pred_text = decoded.replace(question.lower().strip(), "").strip()
            pred_text = normalize_answer(pred_text.split()[0] if pred_text else "")
            
            # Evaluate
            total += 1
            if pred_text == gt_answer:
                correct += 1
                
            results_list.append({
                "image_name": r["image_name"],
                "question": question,
                "answer": gt_answer,
                "prediction": pred_text,
                "is_correct": pred_text == gt_answer,
                "type": "closed"
            })
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"{'='*60}")
    print(f"Closed-Ended Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"{'='*60}")
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results_list
    }


def evaluate_open(model, processor, test_data, device, compute_bleu=True, biobert_eval=None):
    """Evaluate open-ended questions with BLEU and optional BioBERT similarity"""
    total = 0
    correct = 0
    total_bleu = 0.0
    biobert_sum = 0.0
    biobert_strict = 0  # >0.95
    biobert_soft = 0    # >0.85
    smooth = SmoothingFunction().method1
    
    # Store examples for visualization
    results_list = []
    
    with torch.no_grad():
        for r in tqdm(test_data, desc="Open-Ended"):
            # Load image
            img_path = os.path.join(IMAGE_DIR, r["image_name"])
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            
            # Prepare input - Add <image> token explicitly
            # Prepare input
            question = r["question"]
            # FIX: Training script did not use <image> token manually
            prompt = question
            gt_answer = normalize_answer(r["answer"])
            
            # Generate prediction
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            decoded = processor.decode(outputs[0], skip_special_tokens=True).lower().strip()
            
            # Extract prediction
            pred_text = normalize_answer(decoded.replace(question.lower().strip(), "").strip())
            
            # Exact match accuracy
            total += 1
            if pred_text == gt_answer:
                correct += 1
            
            # BLEU score
            if compute_bleu:
                ref_tokens = gt_answer.split()
                hyp_tokens = pred_text.split()
                bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
                total_bleu += bleu
            
            # BioBERT similarity
            if biobert_eval is not None:
                sim = biobert_eval.compute_similarity(pred_text, gt_answer)
                biobert_sum += sim
                if sim > 0.95:
                    biobert_strict += 1
                if sim > 0.85:
                    biobert_soft += 1
            
            results_list.append({
                "image_name": r["image_name"],
                "question": question,
                "answer": gt_answer,
                "prediction": pred_text,
                "is_correct": pred_text == gt_answer,
                "type": "open"
            })
    
    accuracy = correct / total if total > 0 else 0
    avg_bleu = total_bleu / total if total > 0 else 0
    
    stats = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results_list
    }
    
    if compute_bleu:
        stats["bleu"] = avg_bleu
    
    if biobert_eval is not None:
        stats["biobert_avg"] = biobert_sum / total if total > 0 else 0
        stats["biobert_strict"] = biobert_strict
        stats["biobert_soft"] = biobert_soft
    
    print(f"{'='*60}")
    print(f"Open-Ended Results:")
    print(f"  Total: {total}")
    print(f"  Correct (Exact Match): {correct}")
    print(f"  Accuracy: {accuracy:.4f}")
    if compute_bleu:
        print(f"  Average BLEU: {avg_bleu:.4f}")
    if biobert_eval is not None:
        print(f"  BioBERT Avg Similarity: {stats['biobert_avg']:.4f}")
        print(f"  BioBERT Strict (>0.95): {biobert_strict}")
        print(f"  BioBERT Soft (>0.85): {biobert_soft}")
    print(f"{'='*60}")
    
    return stats


def visualize_results(closed_results, open_results, save_dir="results/paligemma/qualitative"):
    """
    Visualize correct/incorrect predictions matching SAN-RAD style.
    Generates summary grids and individual examples.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating qualitative visualizations in {save_dir}...")
    
    def process_category(results, category_name):
        correct = [r for r in results if r['is_correct']]
        incorrect = [r for r in results if not r['is_correct']]
        
        # 1. Generate Individual Examples (Top 6)
        print(f"  Visualizing {category_name} examples...")
        
        for i, r in enumerate(correct[:6]):
            _save_individual_viz(r, os.path.join(save_dir, f"{category_name}_correct_{i+1}.png"), "green")
            
        for i, r in enumerate(incorrect[:6]):
            _save_individual_viz(r, os.path.join(save_dir, f"{category_name}_incorrect_{i+1}.png"), "red")
            
        # 2. Generate Summary Grid (2x6)
        _create_summary_grid(correct[:6], incorrect[:6], save_dir, category_name)
    
    process_category(closed_results, "closed")
    process_category(open_results, "open")
    print("Optimization Complete.")


def _save_individual_viz(result, save_path, color):
    """Save a single example visualization."""
    import matplotlib.pyplot as plt
    
    img_path = os.path.join(IMAGE_DIR, result['image_name'])
    if not os.path.exists(img_path): return
    
    img = Image.open(img_path).convert('RGB')
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    
    # Overlay text
    title = f"Q: {result['question']}\nPred: {result['prediction']}\nGT: {result['answer']}"
    plt.title(title, fontsize=14, color=color, wrap=True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def _create_summary_grid(correct, incorrect, save_dir, prefix):
    """Create 2x6 summary grid matching SAN-RAD style."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    
    # Top row: Correct
    for i in range(6):
        ax = axes[0, i]
        if i < len(correct):
            r = correct[i]
            img_path = os.path.join(IMAGE_DIR, r['image_name'])
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                ax.set_title(f"✓ {r['type'].upper()}\nPred: {r['prediction'][:15]}", fontsize=10, color='green')
            except:
                pass
        ax.axis('off')
        
    # Bottom row: Incorrect
    for i in range(6):
        ax = axes[1, i]
        if i < len(incorrect):
            r = incorrect[i]
            img_path = os.path.join(IMAGE_DIR, r['image_name'])
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                ax.set_title(f"✗ {r['type'].upper()}\nPred: {r['prediction'][:15]}\nGT: {r['answer'][:15]}", fontsize=9, color='red')
            except:
                pass
        ax.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_grid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved {prefix} grid to {save_path}")

