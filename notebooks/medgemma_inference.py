# -*- coding: utf-8 -*-
"""medgemma_inference.ipynb

MedGemma Zero-Shot VQA-RAD Inference Notebook

This notebook evaluates MedGemma-4B on the VQA-RAD test set using zero-shot prompting.
Designed to run on Kaggle/Colab with GPU.

**Instructions**:
1. Upload the VQA-RAD dataset (`VQA_RAD Dataset Public.json` + `VQA_RAD Image Folder`)
2. Set the paths in the **Configuration** cell below
3. Run all cells
"""

from google.colab import drive
drive.mount('/content/drive')

"""## 1. Install Dependencies"""

!pip install -q transformers accelerate bitsandbytes pillow nltk tqdm matplotlib sentence-transformers

"""## 2. Configuration
**Modify these paths according to your environment**
"""

import os

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# =============================================================================

# Path to the VQA-RAD annotation JSON file
INPUT_DIR = "/content/drive/MyDrive/vqa-rad/"

ANNOTATION_PATH = os.path.join(INPUT_DIR, "input/VQA_RAD Dataset Public.json")

# Path to the VQA-RAD image folder
IMAGE_DIR = os.path.join(INPUT_DIR, "input/VQA_RAD Image Folder")

# Model ID (MedGemma 4B Instruct)
MODEL_ID = "google/medgemma-4b-it"

OUTPUT_DIR = os.path.join(INPUT_DIR, "medgemma_qualitative/")

from google.colab import userdata
from huggingface_hub import login
import getpass

# Hugging Face Token (for gated model access)
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)
    print("✅ Loaded HF_TOKEN from Colab secrets")
except Exception as e:
    print(f"⚠️ Could not load HF_TOKEN from Colab secrets: {e}")
    print("👉 Tip: Add 'HF_TOKEN' to the Secrets tab (key icon) on the left of Colab.")

# =============================================================================
print(f"Annotation Path: {ANNOTATION_PATH}")
print(f"Image Directory: {IMAGE_DIR}")
print(f"Model ID: {MODEL_ID}")

"""## 3. Imports"""

import os
import json
import torch
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For saving figures without display
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import nltk
nltk.download('punkt', quiet=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}/")

# Initialize BioBERT Evaluator for semantic similarity
print("\nInitializing BioBERT for semantic evaluation...")
try:
    from sentence_transformers import SentenceTransformer, util
    biobert_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO').to(DEVICE)
    print(f"✅ BioBERT loaded on {DEVICE}")
    USE_BIOBERT = True
except Exception as e:
    print(f"⚠️  BioBERT not available: {e}")
    print("   Continuing without semantic similarity scores")
    biobert_model = None
    util = None
    USE_BIOBERT = False

"""## 4. Load Dataset"""

def load_test_data():
    """Load VQA-RAD test set."""
    with open(ANNOTATION_PATH) as f:
        records = json.load(f)

    # Filter by phrase_type for test set
    test_data = [d for d in records if d.get("phrase_type") in ["test_freeform", "test_para"]]

    closed_test = [r for r in test_data if r.get("answer_type", "").lower() == "closed"]
    open_test = [r for r in test_data if r.get("answer_type", "").lower() == "open"]

    print(f"Test set: {len(test_data)} total ({len(closed_test)} closed, {len(open_test)} open)")
    return test_data, closed_test, open_test

test_data, closed_test, open_test = load_test_data()

"""## 5. Load Model"""

print("Loading MedGemma with 4-bit quantization...")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("✅ Model loaded successfully!")

"""## 6. Helper Functions"""

# Few-shot template for open-ended questions (from MedGemma notebook)
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


def normalize_answer(text):
    """Lowercase, strip, and standardize answer text."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Standardize yes/no
    if text in ["yes", "y", "true", "1"]:
        return "yes"
    if text in ["no", "n", "false", "0"]:
        return "no"
    return text


def generate_closed_prediction(model, processor, image, question):
    """Generate prediction for a closed-ended question."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"Answer the following question with exactly one word (e.g., 'yes', 'no', 'left', 'right', 'ct', 'mri'). Do not write a sentence. Question: {question}"}
        ]
    }]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(DEVICE)

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

    return pred_text


def generate_open_prediction(model, processor, image, question):
    """Generate prediction for an open-ended question."""
    prompt_text = FEW_SHOT_TEMPLATE.format(question=question)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]
    }]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(DEVICE)

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

    return pred_text


def visualize_single_example(image_path, question, prediction, ground_truth, save_path, is_correct):
    """Create visualization for a single example (EXACT SAN format)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Load and display image
    img = Image.open(image_path).convert('RGB')
    ax.imshow(img)
    ax.axis('off')

    # Add text info at BOTTOM (matching SAN format exactly)
    match_color = 'green' if is_correct else 'red'
    info_text = f" Q: {question} \n Prediction: {prediction} | Ground Truth: {ground_truth}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             color=match_color, weight='bold')

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for bottom text
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_grid(correct, incorrect, save_dir, prefix='summary'):
    """Create a grid showing 6 correct and 6 incorrect examples (matches SAN format)."""
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    # Top row: Correct predictions
    for i in range(6):
        if i < len(correct):
            ex = correct[i]
            img = Image.open(ex['image_path']).convert('RGB')
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"✓ Pred: {ex['prediction'][:20]}",
                                fontsize=10, color='green')
        axes[0, i].axis('off')

    # Bottom row: Incorrect predictions
    for i in range(6):
        if i < len(incorrect):
            ex = incorrect[i]
            img = Image.open(ex['image_path']).convert('RGB')
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"✗ Pred: {ex['prediction'][:15]}\nGT: {ex['ground_truth'][:15]}",
                                fontsize=9, color='red')
        axes[1, i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_grid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"{prefix.capitalize()} grid saved to {save_path}")

"""## 7. Evaluate Closed-Ended Questions"""

def evaluate_closed(model, processor, test_data):
    """Evaluate closed-ended (yes/no) questions."""
    total = 0
    correct = 0

    correct_examples = []
    incorrect_examples = []

    for r in tqdm(test_data, desc="Closed-Ended"):
        img_path = os.path.join(IMAGE_DIR, r["image_name"])
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")

        question = r["question"]
        gt_answer = normalize_answer(r["answer"])

        pred_text = generate_closed_prediction(model, processor, image, question)
        pred_text = normalize_answer(pred_text.split()[0] if pred_text else "")

        total += 1
        is_correct = (pred_text == gt_answer)
        if is_correct:
            correct += 1

        example = {
            "image_path": img_path,
            "question": question,
            "prediction": pred_text,
            "ground_truth": gt_answer,
            "is_correct": is_correct
        }

        if is_correct:
            correct_examples.append(example)
        else:
            incorrect_examples.append(example)

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Closed-Ended Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"{'='*60}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples
    }

closed_results = evaluate_closed(model, processor, closed_test)

"""## 8. Evaluate Open-Ended Questions"""

def evaluate_open(model, processor, test_data, compute_bleu=True):
    """Evaluate open-ended questions with BLEU and BioBERT."""
    total = 0
    correct = 0
    total_bleu = 0.0
    total_biobert = 0.0
    biobert_strict = 0
    biobert_soft = 0
    smooth = SmoothingFunction().method1

    correct_examples = []
    incorrect_examples = []

    for r in tqdm(test_data, desc="Open-Ended"):
        img_path = os.path.join(IMAGE_DIR, r["image_name"])
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")

        question = r["question"]
        gt_answer = normalize_answer(r["answer"])

        pred_text = generate_open_prediction(model, processor, image, question)
        pred_text = normalize_answer(pred_text)

        total += 1
        is_correct = (pred_text == gt_answer)
        if is_correct:
            correct += 1

        if compute_bleu:
            ref_tokens = gt_answer.split()
            hyp_tokens = pred_text.split()
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
            total_bleu += bleu

        if USE_BIOBERT and biobert_model is not None:
            pred_emb = biobert_model.encode(pred_text, convert_to_tensor=True)
            gt_emb = biobert_model.encode(gt_answer, convert_to_tensor=True)
            similarity = util.cos_sim(pred_emb, gt_emb).item()
            total_biobert += similarity
            if similarity > 0.95:
                biobert_strict += 1
            if similarity > 0.85:
                biobert_soft += 1

        example = {
            "image_path": img_path,
            "question": question,
            "prediction": pred_text,
            "ground_truth": gt_answer,
            "is_correct": is_correct
        }

        if is_correct:
            correct_examples.append(example)
        else:
            incorrect_examples.append(example)

    accuracy = correct / total if total > 0 else 0
    avg_bleu = total_bleu / total if total > 0 else 0
    avg_biobert = total_biobert / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Open-Ended Results:")
    print(f"  Total: {total}")
    print(f"  Correct (Exact Match): {correct}")
    print(f"  Accuracy: {accuracy:.4f}")
    if compute_bleu:
        print(f"  Average BLEU: {avg_bleu:.4f}")
    if USE_BIOBERT:
        print(f"  BioBERT Avg Similarity: {avg_biobert:.4f}")
        print(f"  BioBERT Strict (>0.95): {biobert_strict}")
        print(f"  BioBERT Soft (>0.85): {biobert_soft}")
    print(f"{'='*60}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "bleu": avg_bleu,
        "biobert_avg": avg_biobert if USE_BIOBERT else None,
        "biobert_strict": biobert_strict if USE_BIOBERT else None,
        "biobert_soft": biobert_soft if USE_BIOBERT else None,
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples
    }

open_results = evaluate_open(model, processor, open_test)

"""## 9. Overall Results"""

total_correct = closed_results["correct"] + open_results["correct"]
total_questions = closed_results["total"] + open_results["total"]
overall_acc = total_correct / total_questions if total_questions > 0 else 0

print(f"\n{'='*60}")
print("MEDGEMMA EVALUATION - FINAL RESULTS")
print(f"{'='*60}")
print(f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_questions})")
print(f"Closed Accuracy: {closed_results['accuracy']:.4f}")
print(f"Open Accuracy: {open_results['accuracy']:.4f}")
print(f"Open BLEU Score: {open_results['bleu']:.4f}")
if USE_BIOBERT and open_results.get('biobert_avg') is not None:
    print(f"BioBERT Avg Similarity: {open_results['biobert_avg']:.4f}")
    print(f"BioBERT Strict (>0.95): {open_results['biobert_strict']}")
    print(f"BioBERT Soft (>0.85): {open_results['biobert_soft']}")
print(f"{'='*60}")

# Save results
results = {
    "closed": {
        "total": closed_results["total"],
        "correct": closed_results["correct"],
        "accuracy": closed_results["accuracy"]
    },
    "open": {
        "total": open_results["total"],
        "correct": open_results["correct"],
        "accuracy": open_results["accuracy"],
        "bleu": open_results["bleu"],
        "biobert_avg": open_results.get("biobert_avg"),
        "biobert_strict": open_results.get("biobert_strict"),
        "biobert_soft": open_results.get("biobert_soft")
    },
    "overall": {
        "total": total_questions,
        "correct": total_correct,
        "accuracy": overall_acc
    }
}

results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {results_path}")

"""## 10. Qualitative Visualizations"""

print("\n" + "="*80)
print("GENERATING QUALITATIVE VISUALIZATIONS (SAN Format)")
print("="*80)

num_examples = 10

print("\nGenerating visualizations for CLOSED-ENDED questions...")
print("  Saving correct predictions...")
for i, ex in enumerate(closed_results['correct_examples'][:num_examples]):
    save_path = os.path.join(OUTPUT_DIR, f'closed_correct_{i+1}.png')
    visualize_single_example(
        ex['image_path'], ex['question'], ex['prediction'],
        ex['ground_truth'], save_path, is_correct=True
    )
print(f"    ✓ Saved {min(num_examples, len(closed_results['correct_examples']))} correct examples")

print("  Saving incorrect predictions...")
for i, ex in enumerate(closed_results['incorrect_examples'][:num_examples]):
    save_path = os.path.join(OUTPUT_DIR, f'closed_incorrect_{i+1}.png')
    visualize_single_example(
        ex['image_path'], ex['question'], ex['prediction'],
        ex['ground_truth'], save_path, is_correct=False
    )
print(f"    ✓ Saved {min(num_examples, len(closed_results['incorrect_examples']))} incorrect examples")

print("\nGenerating visualizations for OPEN-ENDED questions...")
print("  Saving correct predictions...")
for i, ex in enumerate(open_results['correct_examples'][:num_examples]):
    save_path = os.path.join(OUTPUT_DIR, f'open_correct_{i+1}.png')
    visualize_single_example(
        ex['image_path'], ex['question'], ex['prediction'],
        ex['ground_truth'], save_path, is_correct=True
    )
print(f"    ✓ Saved {min(num_examples, len(open_results['correct_examples']))} correct examples")

print("  Saving incorrect predictions...")
for i, ex in enumerate(open_results['incorrect_examples'][:num_examples]):
    save_path = os.path.join(OUTPUT_DIR, f'open_incorrect_{i+1}.png')
    visualize_single_example(
        ex['image_path'], ex['question'], ex['prediction'],
        ex['ground_truth'], save_path, is_correct=False
    )
print(f"    ✓ Saved {min(num_examples, len(open_results['incorrect_examples']))} incorrect examples")

print("\nCreating summary grids...")
create_summary_grid(
    closed_results['correct_examples'][:6],
    closed_results['incorrect_examples'][:6],
    OUTPUT_DIR, 'closed'
)
create_summary_grid(
    open_results['correct_examples'][:6],
    open_results['incorrect_examples'][:6],
    OUTPUT_DIR, 'open'
)

print("\n" + "="*80)
print(f"✅ Qualitative analysis complete! All results saved to {OUTPUT_DIR}/")
print("="*80)
