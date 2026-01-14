"""
BioBERT Semantic Similarity Evaluator

Uses the PubMedBERT model fine-tuned on MS-MARCO for computing
semantic similarity between predicted and ground truth answers.
Based on the MedGemma notebook evaluation approach.
"""
import torch
from typing import Optional


class BioBERTEvaluator:
    """
    BioBERT-based semantic similarity evaluator.
    
    Uses 'pritamdeka/S-PubMedBert-MS-MARCO' for medical domain embeddings.
    Computes cosine similarity between prediction and ground truth embeddings.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the BioBERT evaluator.
        
        Args:
            device: Device to run on ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Lazy import to avoid loading if not needed
        try:
            from sentence_transformers import SentenceTransformer, util
            self.util = util
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for BioBERT evaluation. "
                "Install with: pip install sentence-transformers"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("⏳ Loading BioBERT Embedding Model...")
        self.model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        self.model.to(self.device)
        print(f"✅ BioBERT loaded on {self.device}")
    
    def compute_similarity(self, prediction: str, ground_truth: str) -> float:
        """
        Compute cosine similarity between prediction and ground truth.
        
        Args:
            prediction: Predicted answer text
            ground_truth: Ground truth answer text
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not prediction or not ground_truth:
            return 0.0
        
        # Encode both texts
        pred_embedding = self.model.encode(prediction, convert_to_tensor=True)
        gt_embedding = self.model.encode(ground_truth, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = self.util.cos_sim(pred_embedding, gt_embedding).item()
        
        return similarity
    
    def batch_compute_similarity(self, predictions: list, ground_truths: list) -> list:
        """
        Compute similarity for a batch of predictions.
        
        Args:
            predictions: List of predicted answer texts
            ground_truths: List of ground truth answer texts
            
        Returns:
            List of similarity scores
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        # Handle empty strings
        valid_pairs = []
        indices = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            if pred and gt:
                valid_pairs.append((pred, gt))
                indices.append(i)
        
        if not valid_pairs:
            return [0.0] * len(predictions)
        
        # Batch encode
        preds_text = [p for p, _ in valid_pairs]
        gts_text = [g for _, g in valid_pairs]
        
        pred_embeddings = self.model.encode(preds_text, convert_to_tensor=True)
        gt_embeddings = self.model.encode(gts_text, convert_to_tensor=True)
        
        # Compute similarities
        similarities = [0.0] * len(predictions)
        for i, (idx, (pred_emb, gt_emb)) in enumerate(
            zip(indices, zip(pred_embeddings, gt_embeddings))
        ):
            sim = self.util.cos_sim(pred_emb.unsqueeze(0), gt_emb.unsqueeze(0)).item()
            similarities[idx] = sim
        
        return similarities
