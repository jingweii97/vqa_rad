"""
Qualitative Analysis: Visualize success and failure cases for report.
"""
import torch
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.common.config import DEVICE, IMAGE_DIR, ANNOTATION_PATH, MODEL_SAVE_PATH
from src.san.dataset import load_data, get_question_type_idx, get_organ_idx
from src.san.model import SAN_RAD
from src.common.utils import clean_text
from src.san.visualize import visualize_attention_map

def analyze_qualitative_results(model, test_data, vocab, ans_vocab, num_examples=10, save_dir='results/qualitative'):
    """
    Generate qualitative analysis by showing correct and incorrect predictions.
    Separates open-ended and closed-ended questions.
    
    Args:
        model: Trained SAN model
        test_data: List of test samples
        vocab: Question vocabulary
        ans_vocab: Answer vocabulary
        num_examples: Number of examples to show for each category
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    idx_to_ans = {idx: ans for ans, idx in ans_vocab.items()}
    
    # Separate tracking for open and closed questions
    open_correct = []
    open_incorrect = []
    closed_correct = []
    closed_incorrect = []
    
    print("Analyzing test set predictions...")
    
    # Load test dataset for batch processing
    from torch.utils.data import DataLoader
    from src.san.dataset import VQARadDataset, get_transforms
    
    _, test_transform = get_transforms()
    test_ds = VQARadDataset(test_data, IMAGE_DIR, vocab, ans_vocab, test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for idx, (image, question, label, a_type, q_type_idx, organ_idx) in enumerate(test_loader):
            if idx >= len(test_data):
                break
                
            image = image.to(DEVICE)
            question = question.to(DEVICE)
            q_type_idx = q_type_idx.to(DEVICE)
            organ_idx = organ_idx.to(DEVICE)
            
            # Get prediction and attention
            output, attn_weights = model.forward_with_attention(image, question, q_type_idx, organ_idx)
            pred_idx = output.argmax(dim=1).item()
            label_idx = label.item()
            
            pred_str = idx_to_ans.get(pred_idx, "<unk>")
            gt_str = idx_to_ans.get(label_idx, "<unk>")
            
            example = {
                'idx': idx,
                'image_path': os.path.join(IMAGE_DIR, test_data[idx]['image_name']),
                'question': test_data[idx]['question'],
                'prediction': pred_str,
                'ground_truth': gt_str,
                'answer_type': a_type[0],
                'attention': attn_weights.cpu().numpy(),
                'image_tensor': image,
                'question_tensor': question,
                'q_type_idx': q_type_idx,
                'organ_idx': organ_idx
            }
            
            # Determine if correct (skip if GT is UNK)
            is_correct = (pred_str == gt_str) and (gt_str != "<unk>")
            is_open = str(a_type[0]).upper() == "OPEN"
            
            # Categorize
            if is_open:
                if gt_str != "<unk>":  # Only include if we know the answer
                    if is_correct:
                        open_correct.append(example)
                    else:
                        open_incorrect.append(example)
            else:  # Closed
                if gt_str != "<unk>":
                    if is_correct:
                        closed_correct.append(example)
                    else:
                        closed_incorrect.append(example)
            
            # Stop when we have enough examples for all categories
            if (len(open_correct) >= num_examples and len(open_incorrect) >= num_examples and
                len(closed_correct) >= num_examples and len(closed_incorrect) >= num_examples):
                break
    
    print(f"Open-ended: {len(open_correct)} correct, {len(open_incorrect)} incorrect")
    print(f"Closed-ended: {len(closed_correct)} correct, {len(closed_incorrect)} incorrect")
    
    # Visualize open-ended examples
    print("\nGenerating visualizations for OPEN-ENDED questions...")
    print("  Correct predictions...")
    for i, ex in enumerate(open_correct[:num_examples]):
        save_path = os.path.join(save_dir, f'open_correct_{i+1}.png')
        visualize_attention_map(
            model, ex['image_tensor'], ex['question_tensor'], 
            ex['q_type_idx'], ex['organ_idx'],
            ex['image_path'], ex['question'], 
            ex['prediction'], ex['ground_truth'],
            device=DEVICE, save_path=save_path
        )
    
    print("  Incorrect predictions...")
    for i, ex in enumerate(open_incorrect[:num_examples]):
        save_path = os.path.join(save_dir, f'open_incorrect_{i+1}.png')
        visualize_attention_map(
            model, ex['image_tensor'], ex['question_tensor'],
            ex['q_type_idx'], ex['organ_idx'],
            ex['image_path'], ex['question'], 
            ex['prediction'], ex['ground_truth'],
            device=DEVICE, save_path=save_path
        )
    
    # Visualize closed-ended examples
    print("\nGenerating visualizations for CLOSED-ENDED questions...")
    print("  Correct predictions...")
    for i, ex in enumerate(closed_correct[:num_examples]):
        save_path = os.path.join(save_dir, f'closed_correct_{i+1}.png')
        visualize_attention_map(
            model, ex['image_tensor'], ex['question_tensor'], 
            ex['q_type_idx'], ex['organ_idx'],
            ex['image_path'], ex['question'], 
            ex['prediction'], ex['ground_truth'],
            device=DEVICE, save_path=save_path
        )
    
    print("  Incorrect predictions...")
    for i, ex in enumerate(closed_incorrect[:num_examples]):
        save_path = os.path.join(save_dir, f'closed_incorrect_{i+1}.png')
        visualize_attention_map(
            model, ex['image_tensor'], ex['question_tensor'],
            ex['q_type_idx'], ex['organ_idx'],
            ex['image_path'], ex['question'], 
            ex['prediction'], ex['ground_truth'],
            device=DEVICE, save_path=save_path
        )
    
    # Create summary grids
    create_summary_grid(open_correct[:6], open_incorrect[:6], save_dir, 'open')
    create_summary_grid(closed_correct[:6], closed_incorrect[:6], save_dir, 'closed')
    
    print(f"\nQualitative analysis complete! Results saved to {save_dir}/")

def create_summary_grid(correct, incorrect, save_dir, prefix='summary'):
    """Create a grid showing 6 correct and 6 incorrect examples."""
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    
    # Top row: Correct predictions
    for i, ex in enumerate(correct):
        img = Image.open(ex['image_path']).convert('RGB')
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"✓ {ex['answer_type']}\nPred: {ex['prediction'][:15]}", 
                            fontsize=10, color='green')
        axes[0, i].axis('off')
    
    # Bottom row: Incorrect predictions
    for i, ex in enumerate(incorrect):
        img = Image.open(ex['image_path']).convert('RGB')
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"✗ {ex['answer_type']}\nPred: {ex['prediction'][:15]}\nGT: {ex['ground_truth'][:15]}", 
                            fontsize=9, color='red')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_grid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"{prefix.capitalize()} grid saved to {save_path}")

if __name__ == "__main__":
    # Load data and model
    print("Loading data...")
    train_loader, val_loader, test_data, vocab, ans_vocab = load_data(use_rephrased_augmentation=True)
    
    print("Loading model...")
    model = SAN_RAD(
        vocab_size=len(vocab),
        ans_vocab_size=len(ans_vocab)
    ).to(DEVICE)
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_SAVE_PATH}")
    else:
        print("No trained model found! Please train first.")
        exit(1)
    
    # Run qualitative analysis
    analyze_qualitative_results(model, test_data, vocab, ans_vocab, num_examples=10)
