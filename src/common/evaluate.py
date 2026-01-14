"""
Unified Evaluation Metrics for VQA-RAD

This module provides standardized evaluation metrics for all three models:
SAN-RAD, PaliGemma, and MedGemma.

Metrics computed:
- Accuracy (exact match) for all questions
- BLEU score for open-ended questions
- BioBERT semantic similarity for open-ended questions
"""
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any, Optional


def normalize_answer(answer: str) -> str:
    """Normalize answer text for comparison."""
    if answer is None:
        return ""
    return str(answer).lower().strip()


def compute_exact_match(pred: str, gt: str) -> bool:
    """Check if prediction exactly matches ground truth (normalized)."""
    return normalize_answer(pred) == normalize_answer(gt)


def compute_bleu(pred: str, gt: str) -> float:
    """
    Compute BLEU score between prediction and ground truth.
    Uses smoothing to handle short sentences.
    """
    smooth = SmoothingFunction().method1
    ref_tokens = normalize_answer(gt).split()
    hyp_tokens = normalize_answer(pred).split()
    
    if not hyp_tokens:
        hyp_tokens = ["<empty>"]
    if not ref_tokens:
        return 0.0
        
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)


def compute_all_metrics(
    predictions: List[str],
    ground_truths: List[str],
    answer_types: List[str],
    biobert_evaluator=None
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a set of predictions.
    
    Args:
        predictions: List of predicted answer strings
        ground_truths: List of ground truth answer strings
        answer_types: List of answer types ('open' or 'closed')
        biobert_evaluator: Optional BioBERTEvaluator instance for semantic similarity
        
    Returns:
        Dictionary with comprehensive metrics:
        {
            "closed": {
                "total": int,
                "correct": int,
                "accuracy": float
            },
            "open": {
                "total": int,
                "correct": int,
                "accuracy": float,
                "bleu_avg": float,
                "biobert_avg": float,  # if evaluator provided
                "biobert_strict": int, # matches > 0.95
                "biobert_soft": int    # matches > 0.85
            },
            "overall": {
                "total": int,
                "correct": int,
                "accuracy": float
            },
            "per_sample": [...]  # List of per-sample results
        }
    """
    assert len(predictions) == len(ground_truths) == len(answer_types), \
        "All input lists must have the same length"
    
    # Initialize stats
    stats = {
        "closed": {"total": 0, "correct": 0},
        "open": {"total": 0, "correct": 0, "bleu_sum": 0.0, 
                 "biobert_sum": 0.0, "biobert_strict": 0, "biobert_soft": 0},
        "per_sample": []
    }
    
    for pred, gt, a_type in zip(predictions, ground_truths, answer_types):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)
        is_open = a_type.lower() == "open"
        group = "open" if is_open else "closed"
        
        # Exact match
        is_correct = compute_exact_match(pred, gt)
        
        # Per-sample record
        sample_result = {
            "prediction": pred_norm,
            "ground_truth": gt_norm,
            "answer_type": a_type,
            "correct": is_correct
        }
        
        # Update counts
        stats[group]["total"] += 1
        if is_correct:
            stats[group]["correct"] += 1
        
        # Open-ended specific metrics
        if is_open:
            # BLEU
            bleu = compute_bleu(pred, gt)
            stats["open"]["bleu_sum"] += bleu
            sample_result["bleu"] = bleu
            
            # BioBERT similarity (if evaluator provided)
            if biobert_evaluator is not None:
                sim = biobert_evaluator.compute_similarity(pred_norm, gt_norm)
                stats["open"]["biobert_sum"] += sim
                if sim > 0.95:
                    stats["open"]["biobert_strict"] += 1
                if sim > 0.85:
                    stats["open"]["biobert_soft"] += 1
                sample_result["biobert_similarity"] = sim
        
        stats["per_sample"].append(sample_result)
    
    # Compute averages
    closed_total = stats["closed"]["total"]
    open_total = stats["open"]["total"]
    
    result = {
        "closed": {
            "total": closed_total,
            "correct": stats["closed"]["correct"],
            "accuracy": stats["closed"]["correct"] / max(closed_total, 1)
        },
        "open": {
            "total": open_total,
            "correct": stats["open"]["correct"],
            "accuracy": stats["open"]["correct"] / max(open_total, 1),
            "bleu_avg": stats["open"]["bleu_sum"] / max(open_total, 1)
        },
        "overall": {
            "total": closed_total + open_total,
            "correct": stats["closed"]["correct"] + stats["open"]["correct"],
            "accuracy": (stats["closed"]["correct"] + stats["open"]["correct"]) / 
                        max(closed_total + open_total, 1)
        },
        "per_sample": stats["per_sample"]
    }
    
    # Add BioBERT stats if computed
    if biobert_evaluator is not None and open_total > 0:
        result["open"]["biobert_avg"] = stats["open"]["biobert_sum"] / open_total
        result["open"]["biobert_strict"] = stats["open"]["biobert_strict"]
        result["open"]["biobert_soft"] = stats["open"]["biobert_soft"]
    
    return result


def print_evaluation_report(metrics: Dict[str, Any], model_name: str = "Model"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"📊 {model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Closed-ended
    c = metrics["closed"]
    print(f"\n🔒 CLOSED-ENDED (Yes/No/Choice)")
    print(f"   Total: {c['total']}")
    print(f"   Correct: {c['correct']}")
    print(f"   Accuracy: {c['accuracy']:.4f} ({c['accuracy']*100:.2f}%)")
    
    # Open-ended
    o = metrics["open"]
    print(f"\n🔓 OPEN-ENDED (Descriptive)")
    print(f"   Total: {o['total']}")
    print(f"   Correct (Exact Match): {o['correct']}")
    print(f"   Accuracy: {o['accuracy']:.4f} ({o['accuracy']*100:.2f}%)")
    print(f"   Average BLEU: {o['bleu_avg']:.4f}")
    
    if "biobert_avg" in o:
        print(f"\n   🧬 BioBERT Semantic Similarity:")
        print(f"      Average Score: {o['biobert_avg']:.4f}")
        print(f"      Strict Matches (>0.95): {o['biobert_strict']}")
        print(f"      Soft Matches (>0.85): {o['biobert_soft']}")
    
    # Overall
    ov = metrics["overall"]
    print(f"\n📈 OVERALL")
    print(f"   Total: {ov['total']}")
    print(f"   Correct: {ov['correct']}")
    print(f"   Accuracy: {ov['accuracy']:.4f} ({ov['accuracy']*100:.2f}%)")
    
    print(f"{'='*60}\n")
