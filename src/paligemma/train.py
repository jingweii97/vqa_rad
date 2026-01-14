"""
VLM Training Module for PaliGemma2

Handles QLoRA fine-tuning of PaliGemma2 on VQA-RAD dataset.
"""
import os
import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType

from src.common.config import (
    DEVICE, SEED, MODEL_DIR,
    VLM_MODEL_ID, VLM_LORA_R, VLM_LORA_ALPHA, VLM_LORA_DROPOUT,
    VLM_LEARNING_RATE, VLM_EPOCHS, VLM_GRADIENT_ACCUMULATION,
    VLM_EVAL_STEPS, VLM_SAVE_STEPS, BATCH_SIZE
)


class ValidationLossCallback(TrainerCallback):
    """Print validation loss clearly during training"""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss is not None:
                print(f"\n{'='*60}")
                print(f"📊 VALIDATION at Step {state.global_step}")
                print(f"{'='*60}")
                print(f"   Eval Loss: {eval_loss:.4f}")
                print(f"{'='*60}\n")


def load_paligemma_model(hf_token=None):
    """
    Load quantized PaliGemma2 model with QLoRA adapters.
    
    Args:
        hf_token: HuggingFace token for model access
        
    Returns:
        model, processor
    """
    # Quantization config for 4-bit training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading model: {VLM_MODEL_ID}")
    print(f"👉 Ensure you have accepted the license at: https://huggingface.co/{VLM_MODEL_ID}")
    
    # Load base model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    
    processor = PaliGemmaProcessor.from_pretrained(VLM_MODEL_ID, token=hf_token)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=VLM_LORA_R,
        lora_alpha=VLM_LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],  # Attention layers only
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=VLM_LORA_DROPOUT,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor


def train_paligemma(model, processor, train_data, val_data, collate_fn, output_dir=None, resume_from_checkpoint=None):
    """
    Train VLM model with QLoRA.
    
    Args:
        model: PEFT-wrapped PaliGemma model
        processor: PaliGemma processor (for saving alongside model)
        train_data: List of training samples
        val_data: List of validation samples
        collate_fn: VLMCollator instance
        output_dir: Directory to save checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        trainer: Trained Trainer object
    """
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, "vlm_checkpoints")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Constrained by GPU memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=VLM_GRADIENT_ACCUMULATION,
        gradient_checkpointing=True,
        
        # Training duration
        num_train_epochs=VLM_EPOCHS,
        
        # Learning rate
        learning_rate=VLM_LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Evaluation strategy
        eval_strategy="steps",
        eval_steps=VLM_EVAL_STEPS,
        save_strategy="steps",
        save_steps=VLM_SAVE_STEPS,
        save_total_limit=3,
        
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Performance
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="linear",
        
        # Other
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
        callbacks=[ValidationLossCallback()]
    )
    
    print("🎯 Starting VLM training...\n")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print("="*60 + "\n")
    
    # Train
    if resume_from_checkpoint:
        print(f"📂 Resuming from checkpoint: {resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    # Print summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Final Train Loss: {train_result.training_loss:.4f}")
    print(f"Total Steps: {train_result.global_step}")
    print(f"Training Time: {train_result.metrics.get('train_runtime', 0):.2f}s")
    print("="*60 + "\n")
    
    # Save final model
    save_path = os.path.join(MODEL_DIR, "best_vlm_adapter")
    print(f"💾 Saving best model adapter to: {save_path}")
    trainer.model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"✅ Adapter and processor saved!\n")
    
    # Final evaluation
    print("📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    print("="*60)
    
    return trainer
