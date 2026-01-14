import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.common.config import DEVICE
from src.common.semantic_eval import BioBERTEvaluator

def evaluate(model, dataloader, ans_vocab, model_type='san', return_stats=False, use_biobert=False):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the dataset.
        ans_vocab: Answer vocabulary mapping.
        model_type: 'san' or 'vlm'.
        return_stats: If True, return statistics dictionary for plotting.
        use_biobert: If True, compute BioBERT similarity for open-ended questions.
    """
    # Create reverse vocab: index → answer string
    idx_to_ans = {idx: ans for ans, idx in ans_vocab.items()}
    unk_idx = ans_vocab.get("<unk>", 0)
    
    model.eval()

    # Stats breakdown
    stats = {
        "open": {"correct": 0, "total": 0, "unk_label": 0, "unk_pred": 0, "unk_match": 0, "bleu_sum": 0.0,
                 "biobert_sum": 0.0, "biobert_strict": 0, "biobert_soft": 0},
        "closed": {"correct": 0, "total": 0, "unk_label": 0, "unk_pred": 0, "unk_match": 0},
    }
    
    # BLEU smoothing function
    smooth = SmoothingFunction().method1
    
    # BioBERT evaluator (optional)
    biobert_eval = None
    if use_biobert:
        print("⏳ Initializing BioBERT evaluator...")
        biobert_eval = BioBERTEvaluator(device=DEVICE)
    
    print(f"Starting evaluation (Model Type: {model_type.upper()})...")

    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'san':
                images, questions, labels, a_types, q_type_idx, organ_idx = batch
                images = images.to(DEVICE)
                questions = questions.to(DEVICE)
                labels = labels.to(DEVICE)
                q_type_idx = q_type_idx.to(DEVICE)
                organ_idx = organ_idx.to(DEVICE)

                outputs = model(images, questions, q_type_idx, organ_idx)
                preds = outputs.argmax(dim=1)
                
            elif model_type == 'vlm':
                # TODO: Implement VLM inference
                # images, text_questions, labels_str, a_types = batch
                # outputs = model.generate(images, text_questions)
                # preds = [process_output(o) for o in outputs]
                print("VLM evaluation not implemented.")
                return 

            for i, a_type in enumerate(a_types):
                if model_type == 'san':
                    pred_idx = preds[i].item()
                    label_idx = labels[i].item()
                    
                    pred_str = idx_to_ans.get(pred_idx, "<unk>")
                    actual_str = idx_to_ans.get(label_idx, "<unk>")
                    
                    # Check for UNK bias
                    is_unk_label = (label_idx == unk_idx)
                    is_unk_pred = (pred_idx == unk_idx)
                    is_unk_match = (is_unk_label and is_unk_pred)
                    
                    # A match is truly correct only if it's not an UNK match
                    is_correct = (pred_idx == label_idx) and not is_unk_match
                else:
                    # VLM logic would go here (string matching)
                    is_correct = False 

                # Determine group
                # a_type might be "OPEN", "CLOSED", or lowercase/mixed in dataset
                # Normalize to check
                norm_type = str(a_type).upper()
                group = "open" if "OPEN" in norm_type else "closed"

                stats[group]["total"] += 1
                if is_correct: stats[group]["correct"] += 1
                if model_type == 'san':
                    if is_unk_label: stats[group]["unk_label"] += 1
                    if is_unk_pred: stats[group]["unk_pred"] += 1
                    if is_unk_match: stats[group]["unk_match"] += 1
                    
                    # Compute BLEU for open-ended questions
                    if group == "open" and not is_unk_label:
                        ref_tokens = actual_str.split()
                        hyp_tokens = pred_str.split()
                        if hyp_tokens and ref_tokens:
                            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
                            stats["open"]["bleu_sum"] += bleu
                        
                        # Compute BioBERT similarity for open-ended questions
                        if biobert_eval is not None:
                            sim = biobert_eval.compute_similarity(pred_str, actual_str)
                            stats["open"]["biobert_sum"] += sim
                            if sim > 0.95:
                                stats["open"]["biobert_strict"] += 1
                            if sim > 0.85:
                                stats["open"]["biobert_soft"] += 1
                
                # Verbose print (optional, keeping it minimal for bulk eval, 
                # or uncomment to see every prediction like in original script)
                # print(f"[{a_type}] Pred: {pred_str} | Act: {actual_str} | Corr: {is_correct}")

    # Print summary statistics
    print(f"\n{'='*60}")
    for k in stats:
        total = stats[k]["total"]
        if total == 0: continue
        
        # Valid total: exclude samples where we don't even know the answer (UNK label)
        valid_total = total - stats[k]["unk_label"]
        raw_acc = stats[k]["correct"] / total
        
        # Adjusted accuracy: Correct matches / Valid total
        adj_acc = stats[k]["correct"] / max(valid_total, 1)
        
        print(f"{k.capitalize()} Answers:")
        print(f"  Raw Accuracy (with UNK matches suppressed): {raw_acc:.4f} ({stats[k]['correct']}/{total})")
        print(f"  Adjusted Accuracy (excluding UNK labels): {adj_acc:.4f} ({stats[k]['correct']}/{valid_total})")
        if model_type == 'san':
            print(f"  UNK labels: {stats[k]['unk_label']} ({stats[k]['unk_label']/max(total,1):.1%} of test set)")
            print(f"  UNK predictions: {stats[k]['unk_pred']}")
            print(f"  UNK matches (suppressed): {stats[k]['unk_match']}")
            # Print BLEU and BioBERT for open-ended questions
            if k == "open" and "bleu_sum" in stats[k]:
                bleu_avg = stats[k]["bleu_sum"] / max(valid_total, 1)
                print(f"  Average BLEU Score: {bleu_avg:.4f}")
            if k == "open" and biobert_eval is not None:
                biobert_avg = stats[k]["biobert_sum"] / max(valid_total, 1)
                print(f"  BioBERT Avg Similarity: {biobert_avg:.4f}")
                print(f"  BioBERT Strict (>0.95): {stats[k]['biobert_strict']}")
                print(f"  BioBERT Soft (>0.85): {stats[k]['biobert_soft']}")
        print("-" * 30)

    overall_correct = sum(v["correct"] for v in stats.values())
    overall_total = sum(v["total"] for v in stats.values())
    overall_unk_label = sum(v["unk_label"] for v in stats.values())
    
    overall_raw_acc = overall_correct / max(overall_total, 1)
    overall_adj_acc = overall_correct / max(overall_total - overall_unk_label, 1)

    print(f"Overall Raw Accuracy: {overall_raw_acc:.4f}")
    print(f"Overall Adjusted Accuracy: {overall_adj_acc:.4f}")
    print(f"{'='*60}")
    
    if return_stats:
        return stats

