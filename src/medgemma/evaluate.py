"""
MedGemma Zero-Shot Evaluation Module

Evaluates MedGemma-4B on VQA-RAD test set using zero-shot prompting.
Based on the MedGemma_VQA_Rad.ipynb notebook.
"""
import os
import torch
from PIL import Image
from tqdm import tqdm

from src.common.config import IMAGE_DIR
from src.common.data_loader import get_test_data_only
from src.common.evaluate import compute_all_metrics, print_evaluation_report
from src.common.semantic_eval import BioBERTEvaluator


# Few-shot template for open-ended questions (from notebook)
FEW_SHOT_TEMPLATE = """Provide a short, concise answer to this radiology question.
Only answer in short words or phrases (e.g., 'pancreas', 'free air', 'diffuse', 'posterior to the appendix', 'left hepatic lobe')
Do not write a sentence. Follow these examples:

Q: What modality is shown?
A: x-ray

Q: Where is the opacity located?
A: right upper lobe

Q: How would you describe the spleen abnormality?
A: hypodense lesion

Q: What is the abnormality seen?
A: pleural effusion

Q: What is the plane of this image?
A: axial

Q: How big is the liver lesion?
A: 5 cm

Now, answer this question for the image provided:
Q: {question}
A:"""


def load_medgemma_model(model_id="google/medgemma-4b-it", device="cuda"):
    """
    Load MedGemma model with 4-bit quantization.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load on
        
    Returns:
        model, processor
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Loading MedGemma: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("✅ Model loaded successfully!")
    
    return model, processor


def evaluate_medgemma(
    model, 
    processor, 
    test_data=None,
    device="cuda",
    use_biobert=True
):
    """
    Evaluate MedGemma on VQA-RAD test set.
    
    Args:
        model: MedGemma model
        processor: MedGemma processor
        test_data: Test data (loads automatically if None)
        device: Device to run on
        use_biobert: Whether to compute BioBERT similarity
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load test data if not provided
    if test_data is None:
        test_data, closed_test, open_test = get_test_data_only()
    else:
        closed_test = [r for r in test_data if r.get("answer_type", "").lower() == "closed"]
        open_test = [r for r in test_data if r.get("answer_type", "").lower() == "open"]
    
    print(f"\n{'='*60}")
    print("MEDGEMMA ZERO-SHOT EVALUATION")
    print(f"{'='*60}")
    print(f"  Closed questions: {len(closed_test)}")
    print(f"  Open questions: {len(open_test)}")
    print(f"  Total: {len(test_data)}")
    print(f"{'='*60}\n")
    
    # Initialize BioBERT evaluator if needed
    biobert_eval = BioBERTEvaluator(device=device) if use_biobert else None
    
    # Collect predictions
    predictions = []
    ground_truths = []
    answer_types = []
    
    model.eval()
    
    # Evaluate closed-ended questions
    # Evaluate closed-ended questions
    print("🔒 Evaluating CLOSED-ENDED questions...")
    closed_results_list = []
    
    for item in tqdm(closed_test, desc="Closed-Ended"):
        pred, res_dict = _generate_closed(model, processor, item, device)
        predictions.append(pred)
        ground_truths.append(str(item["answer"]).lower().strip())
        answer_types.append("closed")
        closed_results_list.append(res_dict)
    
    # Evaluate open-ended questions
    print("\n🔓 Evaluating OPEN-ENDED questions...")
    open_results_list = []
    
    for item in tqdm(open_test, desc="Open-Ended"):
        pred, res_dict = _generate_open(model, processor, item, device)
        predictions.append(pred)
        ground_truths.append(str(item["answer"]).lower().strip())
        answer_types.append("open")
        open_results_list.append(res_dict)
    
    # Compute unified metrics
    metrics = compute_all_metrics(
        predictions, ground_truths, answer_types, 
        biobert_evaluator=biobert_eval
    )
    
    # Print report
    print_evaluation_report(metrics, model_name="MedGemma Zero-Shot")
    
    # Generate visualizations
    try:
        visualize_results(closed_results_list, open_results_list)
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    
    return metrics


def _generate_closed(model, processor, item, device):
    """Generate prediction for closed-ended question."""
    try:
        img_path = os.path.join(IMAGE_DIR, item['image_name'])
        image = Image.open(img_path).convert("RGB")
    except:
        return "", {}
    
    question = item['question']
    gt_answer = str(item["answer"]).lower().strip()
    
    # Strict prompt for closed-ended
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"Answer the following question with exactly one word (e.g., 'yes', 'no', 'left', 'right', 'ct', 'mri'). Do not write a sentence. Question: {question}"}
        ]
    }]
    
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )
    
    # Decode and clean
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    pred_text = "".join([c for c in pred_text if c.isalnum() or c.isspace()])
    
    res_dict = {
        "image_name": item['image_name'],
        "question": question,
        "answer": gt_answer,
        "prediction": pred_text,
        "is_correct": pred_text == gt_answer,
        "type": "closed"
    }
    
    return pred_text, res_dict


def _generate_open(model, processor, item, device):
    """Generate prediction for open-ended question."""
    try:
        img_path = os.path.join(IMAGE_DIR, item['image_name'])
        image = Image.open(img_path).convert("RGB")
    except:
        return "", {}
    
    question = item['question']
    gt_answer = str(item["answer"]).lower().strip()
    prompt_text = FEW_SHOT_TEMPLATE.format(question=question)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]
    }]
    
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode and clean
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    pred_text = pred_text.replace("a:", "").strip()
    
    res_dict = {
        "image_name": item['image_name'],
        "question": question,
        "answer": gt_answer,
        "prediction": pred_text,
        "is_correct": pred_text == gt_answer,
        "type": "open"
    }
    
    return pred_text, res_dict


def visualize_results(closed_results, open_results, save_dir="results/medgemma/qualitative"):
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
