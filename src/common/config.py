import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up two levels: src/common/ -> src/ -> vqa-rad/
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# @input convention
DATA_DIR = os.path.join(PROJECT_ROOT, "input")
IMAGE_DIR = os.path.join(DATA_DIR, "VQA_RAD Image Folder")
ANNOTATION_PATH = os.path.join(DATA_DIR, "VQA_RAD Dataset Public.json")

# @model convention
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_san.pth")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 500
LEARNING_RATE = 5e-4
EPOCHS = 100
PATIENCE = 10
EMBED_DIM = 300
LSTM_HIDDEN = 512
ATTENTION_DIM = 512
CNN_DIM = 2048

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 42

# VLM Configuration (PaliGemma2)
VLM_MODEL_ID = "google/paligemma2-3b-pt-224"
VLM_LORA_R = 32
VLM_LORA_ALPHA = 128
VLM_LORA_DROPOUT = 0.1
VLM_LEARNING_RATE = 5e-5
VLM_EPOCHS = 3
VLM_GRADIENT_ACCUMULATION = 4
VLM_EVAL_STEPS = 500
VLM_SAVE_STEPS = 500

# MedGemma Configuration (Zero-Shot)
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"

