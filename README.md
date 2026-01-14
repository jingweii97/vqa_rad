# VQA-RAD Project

Medical Visual Question Answering on the VQA-RAD Dataset. This repository implements three approaches:

1. **SAN-RAD**: A Stacked Attention Network (CNN-Attention) baseline with ResNet50 and bidirectional LSTM
2. **PaliGemma2**: QLoRA fine-tuned Vision-Language Model for Medical VQA
3. **MedGemma**: Zero-shot evaluation using Google's MedGemma-4B-IT

## Project Structure

```
vqa_rad/
├── main_san.py                # Entry point for SAN-RAD model
├── main_paligemma.py          # Entry point for PaliGemma2 model
├── main_medgemma.py           # Entry point for MedGemma zero-shot
├── requirements.txt           # Python dependencies
├── input/                     # VQA-RAD dataset
│   ├── VQA_RAD Image Folder/  # Medical images
│   └── VQA_RAD Dataset Public.json
├── model/                     # Trained model checkpoints
│   ├── best_san.pth           # SAN-RAD model
│   └── paligemma_best_model/  # PaliGemma2 adapter
├── results/                   # Evaluation outputs
│   └── san/                   # SAN-RAD results
├── notebooks/                 # Inference notebooks
│   ├── paligemma_inference.ipynb
│   └── medgemma_inference.ipynb
└── src/                       # Source code
    ├── common/                # Shared utilities
    │   ├── config.py          # Configuration (paths, hyperparameters)
    │   ├── data_loader.py     # Data loading utilities
    │   ├── evaluate.py        # Shared evaluation logic
    │   ├── preprocess.py      # Text preprocessing
    │   ├── semantic_eval.py   # BioBERT semantic similarity
    │   └── utils.py           # Shared utility functions
    ├── san/                   # SAN-RAD model
    │   ├── model.py           # SAN architecture (ResNet50 + Bi-LSTM + Attention)
    │   ├── dataset.py         # VQA-RAD dataset for SAN
    │   ├── train.py           # SAN training loop
    │   ├── evaluate.py        # SAN evaluation
    │   └── visualize.py       # Attention visualization
    ├── paligemma/             # PaliGemma2 model
    │   ├── dataset.py         # VLM dataset preparation
    │   ├── train.py           # QLoRA fine-tuning
    │   └── evaluate.py        # VLM evaluation
    ├── medgemma/              # MedGemma model
    │   └── evaluate.py        # Zero-shot evaluation
    └── script/                # Analysis scripts
        └── analyze_qualitative.py  # Attention map visualization
```

## Setup

1. **Environment**: Python 3.8+ recommended
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Data**: Place the VQA-RAD dataset in the `input/` directory:
   - `input/VQA_RAD Image Folder/` - Medical images
   - `input/VQA_RAD Dataset Public.json` - Annotations
4. **HuggingFace Token**: Required for PaliGemma2 and MedGemma models. Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

## Usage

### SAN-RAD Baseline

**Training:**
```bash
python main_san.py --train
```

**Evaluation:**
```bash
python main_san.py --eval --checkpoint model/best_san.pth
```

**With BioBERT semantic similarity:**
```bash
python main_san.py --eval --checkpoint model/best_san.pth --use_biobert
```

### PaliGemma2 (QLoRA Fine-tuning)

**Training:**
```bash
python main_paligemma.py --train --hf_token YOUR_HF_TOKEN
```

**Resume training:**
```bash
python main_paligemma.py --train --hf_token YOUR_HF_TOKEN --resume model/vlm_checkpoints/checkpoint-XXX
```

**Evaluation:**
```bash
python main_paligemma.py --eval --hf_token YOUR_HF_TOKEN --checkpoint model/best_paligemma_adapter
```

### MedGemma (Zero-Shot)

**Evaluation:**
```bash
python main_medgemma.py --hf_token YOUR_HF_TOKEN
```

**Skip BioBERT evaluation:**
```bash
python main_medgemma.py --hf_token YOUR_HF_TOKEN --no_biobert
```

## Models

### SAN-RAD Architecture
- **Image Encoder**: ResNet50 (layer4 unfrozen) → 7×7×2048 spatial features
- **Question Encoder**: Bidirectional LSTM → 2×512-dim hidden states
- **Metadata Embeddings**: Question type + Organ embeddings concatenated to query
- **Attention**: 2-layer stacked attention mechanism
- **Classifier**: MLP with dropout (0.5)

### PaliGemma2 Configuration
- **Base Model**: `google/paligemma2-3b-pt-224`
- **Fine-tuning**: QLoRA with 4-bit quantization (NF4)
- **LoRA Config**: r=32, alpha=128, target_modules=[q_proj, v_proj]
- **Training**: 3 epochs, lr=5e-5, gradient accumulation=4

### MedGemma Configuration
- **Model**: `google/medgemma-4b-it`
- **Inference**: Zero-shot with few-shot prompt template
- **Quantization**: 4-bit (NF4)

## Evaluation Metrics

- **Closed-ended**: Exact match accuracy (yes/no questions)
- **Open-ended**: 
  - Exact match accuracy
  - BLEU score
  - BioBERT semantic similarity (optional)
- **Overall**: Combined accuracy across question types

## Results

Results are saved to `results/` directory:
- `metrics.json` - Evaluation metrics
- `training_curves.png` - Loss/accuracy curves
- `test_accuracy_breakdown.png` - Per-category metrics
- `qualitative/` - Attention map visualizations

## Reproducibility

Global random seeds are set in `src/common/config.py` and applied via `src/common/utils.py` to ensure reproducible results. Default seed: 42.

## License

This project uses the VQA-RAD dataset. Please cite the original paper if using this dataset.

## Acknowledgments

- VQA-RAD Dataset: [Lau et al. (2018)](https://www.nature.com/articles/sdata2018251)
- PaliGemma: Google DeepMind
- MedGemma: Google Research
