"""
Shared preprocessing utilities for VQA-RAD dataset
"""

def expand_with_rephrased_questions(data_list):
    """
    Expand dataset by creating additional samples from rephrased questions.
    
    For each item in data_list, if it has a non-NULL question_rephrase field,
    create a duplicate sample with the rephrased question.
    
    Args:
        data_list: List of data dictionaries
        
    Returns:
        Expanded list with original + rephrased question samples
    """
    expanded = []
    rephrased_count = 0
    
    for item in data_list:
        # Always add the original question
        expanded.append(item)
        
        # Check if there's a valid rephrase (not "NULL" string or None)
        rephrase = item.get("question_rephrase")
        if rephrase and rephrase != "NULL" and rephrase.strip():
            # Create a new sample with the rephrased question
            # Copy the original item to preserve all metadata
            rephrased_item = item.copy()
            rephrased_item["question"] = rephrase
            expanded.append(rephrased_item)
            rephrased_count += 1
    
    print(f"Data augmentation: Added {rephrased_count} rephrased questions. Total: {len(expanded)} (from {len(data_list)})")
    return expanded
