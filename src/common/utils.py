import re
import torch
import numpy as np
import random
from collections import Counter

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def clean_text(text):
    """Convert to lowercase and remove punctuations."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def build_vocab(questions, min_freq=1):
    """Build vocabulary from a list of questions."""
    counter = Counter()
    for q in questions:
        tokens = clean_text(q).split()
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode_question(question, vocab, max_len=30):
    """Encode a question string into a list of indices."""
    tokens = clean_text(question).split()
    encoded = [vocab.get(t, vocab["<unk>"]) for t in tokens]

    if len(encoded) < max_len:
        encoded += [vocab["<pad>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return encoded

def build_answer_vocab(answers, min_freq=1):
    """
    Build answer vocabulary from a list of answers.
    
    Instead of using top_k which cuts off rare but valid answers,
    we include all answers that appear at least min_freq times.
    This ensures better coverage of the answer space.
    """
    counter = Counter([clean_text(a) for a in answers])
    
    ans_vocab = {}
    for ans, count in counter.items():
        if count >= min_freq:
            ans_vocab[ans] = len(ans_vocab)
    
    # Add unknown answer for truly unseen answers
    if "<unk>" not in ans_vocab:
        ans_vocab["<unk>"] = len(ans_vocab)
    
    return ans_vocab
