"""
Unified Data Loading for VQA-RAD

This module provides a single source of truth for loading and splitting
the VQA-RAD dataset, ensuring consistency across all models (SAN-RAD,
PaliGemma, MedGemma).
"""
import os
import json
from sklearn.model_selection import train_test_split
from src.common.config import ANNOTATION_PATH, SEED
from src.common.preprocess import expand_with_rephrased_questions


def load_vqa_rad_raw():
    """
    Load raw VQA-RAD annotations from JSON.
    
    Returns:
        List[dict]: All records from the dataset
    """
    with open(ANNOTATION_PATH) as f:
        return json.load(f)


def load_vqa_rad_splits(use_augmentation=True, verbose=True):
    """
    Load VQA-RAD dataset with standard train/val/test splits.
    
    This function ensures ALL models use identical data splits:
    - Train/Val: 85/15 split with stratification by answer_type
    - Augmentation: Question rephrasing applied BEFORE split
    - Random seed: 42 (from config)
    
    Args:
        use_augmentation: If True, expand with rephrased questions
        verbose: If True, print split statistics
        
    Returns:
        train_data: List of training samples (dicts)
        val_data: List of validation samples (dicts)
        test_data: List of test samples (dicts)
    """
    records = load_vqa_rad_raw()
    
    # Filter by phrase_type
    train_data_all = [d for d in records if d.get("phrase_type") in ["freeform", "para"]]
    test_data = [d for d in records if d.get("phrase_type") in ["test_freeform", "test_para"]]
    
    # Split with stratification (85/15 to match experiment setup)
    if verbose:
        print("Splitting train/val with stratification...")
    
    train_data, val_data = train_test_split(
        train_data_all,
        test_size=0.15,
        random_state=SEED,
        stratify=[d.get("answer_type") for d in train_data_all]
    )

    # Calculate pre-augmentation stats
    if verbose:
        train_open_pre = sum(1 for d in train_data if d.get("answer_type", "").lower() == "open")
        train_closed_pre = len(train_data) - train_open_pre
        print(f"  Train (Pre-Aug): {len(train_data)} total ({train_open_pre} open, {train_closed_pre} closed)")

    # Apply augmentation AFTER split (Correct way to prevent leakage)
    if use_augmentation:
        if verbose:
            print("Applying question rephrasing augmentation to TRAINING set...")
        train_data = expand_with_rephrased_questions(train_data)
    
    if verbose:
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Print breakdown by answer type
        train_open = sum(1 for d in train_data if d.get("answer_type", "").lower() == "open")
        train_closed = len(train_data) - train_open
        
        val_open = sum(1 for d in val_data if d.get("answer_type", "").lower() == "open")
        val_closed = len(val_data) - val_open
        
        test_open = sum(1 for d in test_data if d.get("answer_type", "").lower() == "open")
        test_closed = len(test_data) - test_open
        
        print(f"  Train (Augmented): {train_open} open, {train_closed} closed")
        print(f"  Val:   {val_open} open, {val_closed} closed")
        print(f"  Test:  {test_open} open, {test_closed} closed")
    
    return train_data, val_data, test_data


def get_test_data_only():
    """
    Load only the test set (for zero-shot evaluation).
    
    Returns:
        test_data: List of test samples (dicts)
    """
    records = load_vqa_rad_raw()
    test_data = [d for d in records if d.get("phrase_type") in ["test_freeform", "test_para"]]
    
    # Split by answer type for convenience
    closed_test = [r for r in test_data if r.get("answer_type", "").lower() == "closed"]
    open_test = [r for r in test_data if r.get("answer_type", "").lower() == "open"]
    
    print(f"Test set: {len(test_data)} total ({len(closed_test)} closed, {len(open_test)} open)")
    
    return test_data, closed_test, open_test
