"""
VLM Dataset Module for PaliGemma2

Provides VLM-specific data loading and collation functions.
"""
import os
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.common.config import (
    DATA_DIR, ANNOTATION_PATH, IMAGE_DIR, SEED, 
    BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)
from src.common.preprocess import expand_with_rephrased_questions
from src.common.data_loader import load_vqa_rad_splits


def normalize_answer(answer):
    """Normalize answer for consistency"""
    return answer.lower().strip()


class PaliGemmaCollator:
    """Custom collator for VLM (PaliGemma2) training"""
    
    def __init__(self, processor, image_dir):
        self.processor = processor
        self.image_dir = image_dir
        
    def __call__(self, examples):
        images = []
        texts = []
        
        for example in examples:
            # 1. Load Image
            img_name = example['image_name']
            image_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load {image_path}, using blank image")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            images.append(image)
            
            # 2. Simple prompt format - question + answer
            question = example['question']
            answer = normalize_answer(str(example['answer']))
            full_text = f"{question} {answer}"
            texts.append(full_text)
        
        # 3. Tokenize
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 4. Create Labels & Masking
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        # PaliGemma 224x224 expands <image> token into 256 tokens
        image_token_len = 256
        
        for i, example in enumerate(examples):
            # Calculate question portion to mask
            question = example['question']
            question_tokens = self.processor.tokenizer.encode(question + " ", add_special_tokens=False)
            question_len = len(question_tokens)
            
            # Mask: [Image Tokens (256)] + [Question Tokens]
            mask_len = min(image_token_len + question_len, labels.shape[1])
            labels[i, :mask_len] = -100
        
        # 5. Handle Padding
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        
        inputs["labels"] = labels
        return inputs


def load_paligemma_data(processor):
    """
    Load VQA-RAD data for VLM training.
    
    Uses the unified data_loader for consistent splits across all models.
    
    Returns:
        train_data, val_data, test_data, collate_fn
    """
    # Use unified data loader for consistent splits
    train_data, val_data, test_data = load_vqa_rad_splits(
        use_augmentation=True,
        verbose=True
    )
    
    # Create collator (PaliGemma-specific)
    collate_fn = PaliGemmaCollator(processor, IMAGE_DIR)
    
    # Note: We return raw data lists, not DataLoaders
    # The HuggingFace Trainer will handle batching
    return train_data, val_data, test_data, collate_fn
