# VQA-RAD Project

This repository contains the implementation of a SAN-inspired CNN-Attention baseline for Medical Visual Question Answering on the VQA-RAD dataset. It is designed to be modular and supports future integration of VLM-based models (e.g., PaliGemma).

## Project Structure

```
vqa-rad-project/
├── requirements.txt           # Python dependencies
├── src/
│   ├── config.py              # Configuration (paths, hyperparameters)
│   ├── dataset.py             # Data loading and preprocessing
│   ├── model_san.py           # SAN baseline model definition
│   ├── model_vlm.py           # VLM (PaliGemma) placeholder
│   ├── train_san.py           # SAN training loop
│   ├── train_vlm.py           # VLM training placeholder
│   ├── evaluate.py            # Evaluation logic
│   └── utils.py               # Shared utility functions
├── main_san.py                # Entry point for SAN model
├── main_vlm.py                # Entry point for VLM model
└── results/                   # Directory for outputs
```

## Setup

1. **Environment**: Ensure you have Python installed.
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Data**: The VQA-RAD dataset (images and JSON) is expected to be in the `input/` directory relative to the project root.
   - `input/VQA_RAD Image Folder/`
   - `input/VQA_RAD Dataset Public.json`

## Usage

### Training the SAN Baseline

To train the SAN model from scratch:

```bash
python main_san.py --train
```

The best model will be saved to `model/best_san.pth` (configurable in `src/config.py`).

### Evaluating the SAN Baseline

To evaluate a trained model on the test set:

```bash
python main_san.py --eval --checkpoint model/best_san.pth
```

### VLM (PaliGemma) [Work in Progress]

The VLM integration is currently a placeholder. You can run the stubs:

```bash
python main_vlm.py --train
```

## Reproducibility

Global random seeds are set in `src/config.py` and applied via `src/utils.py` to ensure reproducible results.
