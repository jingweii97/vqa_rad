import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.common.utils import clean_text, encode_question, build_vocab, build_answer_vocab
from src.common.config import IMAGE_DIR, ANNOTATION_PATH, BATCH_SIZE, VAL_BATCH_SIZE, SEED
from src.common.preprocess import expand_with_rephrased_questions
from src.common.data_loader import load_vqa_rad_splits

# Question Type mapping (11 primary types found in dataset)
QUESTION_TYPES = {
    'PRES': 0,      # Presence
    'ABN': 1,       # Abnormality
    'POS': 2,       # Position
    'MODALITY': 3,  # Imaging modality
    'PLANE': 4,     # Imaging plane
    'SIZE': 5,      # Size
    'ATTRIB': 6,    # Attribute
    'COLOR': 7,     # Color
    'COUNT': 8,     # Count
    'OTHER': 9,     # Other
    'UNK': 10       # Unknown/combined types
}
NUM_QUESTION_TYPES = len(QUESTION_TYPES)

# Image Organ mapping (3 types in dataset)
IMAGE_ORGANS = {
    'HEAD': 0,
    'CHEST': 1,
    'ABD': 2,
    'UNK': 3
}
NUM_IMAGE_ORGANS = len(IMAGE_ORGANS)

def get_question_type_idx(qtype_str):
    """Map question type string to index, handling compound types."""
    if qtype_str is None:
        return QUESTION_TYPES['UNK']
    # Handle compound types like "PRES, POS" by taking the first
    primary_type = qtype_str.split(',')[0].strip().upper()
    return QUESTION_TYPES.get(primary_type, QUESTION_TYPES['UNK'])

def get_organ_idx(organ_str):
    """Map image organ string to index."""
    if organ_str is None:
        return IMAGE_ORGANS['UNK']
    return IMAGE_ORGANS.get(organ_str.upper(), IMAGE_ORGANS['UNK'])

class VQARadDataset(Dataset):
    def __init__(self, data, image_dir, vocab, answer_vocab, transform):
        self.data = data
        self.image_dir = image_dir
        self.vocab = vocab
        self.ans_vocab = answer_vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get data instance
        sample = self.data[idx]

        # Image
        image_name = sample["image_name"]
        image_path = os.path.join(self.image_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
             raise FileNotFoundError(f"Image not found at {image_path}")
             
        image = self.transform(image)

        # Question
        question = clean_text(sample.get("question", ""))
        question = encode_question(question, self.vocab)
        question = torch.tensor(question, dtype=torch.long)

        # Answer
        if "answer" in sample:
            answer_text = clean_text(sample["answer"])
            answer = self.ans_vocab.get(answer_text, self.ans_vocab["<unk>"])
            answer = torch.tensor(answer, dtype=torch.long)
        else:
            answer = torch.tensor(-1, dtype=torch.long)

        # Answer type (for evaluation)
        a_type = sample.get('answer_type', 'UNKNOWN')
        
        # NEW: Question type index
        q_type_idx = get_question_type_idx(sample.get('question_type'))
        q_type_idx = torch.tensor(q_type_idx, dtype=torch.long)
        
        # NEW: Image organ index
        organ_idx = get_organ_idx(sample.get('image_organ'))
        organ_idx = torch.tensor(organ_idx, dtype=torch.long)

        return image, question, answer, a_type, q_type_idx, organ_idx

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

# expand_with_rephrased_questions moved to src.common.preprocess

def load_data(use_rephrased_augmentation=True):
    """
    Load data, builds vocabs, and returns dataloaders and vocabs.
    
    Uses the unified data_loader for consistent splits across all models.
    
    Args:
        use_rephrased_augmentation: If True, expand training data with rephrased questions
    """
    # Use unified data loader for consistent splits
    train_data, val_data, test_data = load_vqa_rad_splits(
        use_augmentation=use_rephrased_augmentation,
        verbose=True
    )
    
    # CRITICAL: Build vocabularies from ALL training data (train + val combined)
    # This matches the original behavior where vocab was built BEFORE splitting
    print("Building vocabularies from all training data (train+val)...")
    train_val_combined = train_data + val_data
    all_train_questions = [d["question"] for d in train_val_combined]
    all_train_answers = [d["answer"] for d in train_val_combined]
    
    vocab = build_vocab(all_train_questions, min_freq=1)
    ans_vocab = build_answer_vocab(all_train_answers, min_freq=1)
    
    print(f"  Question vocab size: {len(vocab)}")
    print(f"  Answer vocab size: {len(ans_vocab)}")

    train_transform, test_transform = get_transforms()

    # Create DataLoaders (SAN-specific)
    train_ds = VQARadDataset(train_data, IMAGE_DIR, vocab, ans_vocab, train_transform)
    val_ds = VQARadDataset(val_data, IMAGE_DIR, vocab, ans_vocab, test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_data, vocab, ans_vocab
