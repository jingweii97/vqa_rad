import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from src.common.config import DEVICE, PATIENCE, EPOCHS, MODEL_SAVE_PATH, LEARNING_RATE

def train_san(model, train_loader, val_loader, ans_vocab):
    """
    Train the SAN model and return training history for visualization.
    """
    # Identify the <unk> index
    unk_idx = ans_vocab.get("<unk>", 0)
    
    # Tell the loss function: "Do not count predictions of <unk> towards the loss"
    criterion = nn.CrossEntropyLoss(ignore_index=unk_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_epoch = 0
    trigger_times = 0
    
    # Track metrics for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs': [],
        'best_epoch': 0  # Will be updated when best model is saved
    }
    
    # Check if existing model exists to resume
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found saved model at {MODEL_SAVE_PATH}. Loading weights...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print("Resuming training...")

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, questions, labels, _, q_type_idx, organ_idx) in enumerate(train_loader):
            images = images.to(DEVICE)
            questions = questions.to(DEVICE)
            labels = labels.to(DEVICE)
            q_type_idx = q_type_idx.to(DEVICE)
            organ_idx = organ_idx.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, questions, q_type_idx, organ_idx)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            # Track training accuracy
            preds = outputs.argmax(dim=1)
            valid_mask = (labels != unk_idx) # Ignore UNK for accuracy tracking
            train_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
            train_total += valid_mask.sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / max(train_total, 1)

        # Validation Loop
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, questions, labels, _, q_type_idx, organ_idx in val_loader:
                images = images.to(DEVICE)
                questions = questions.to(DEVICE)
                labels = labels.to(DEVICE)
                q_type_idx = q_type_idx.to(DEVICE)
                organ_idx = organ_idx.to(DEVICE)
                
                outputs = model(images, questions, q_type_idx, organ_idx)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                valid_mask = (labels != unk_idx)
                val_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                val_total += valid_mask.sum().item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / max(val_total, 1)
        
        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2%}) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2%})")

        # Step the learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Early Stopping and Model Saving
        if avg_val_loss < best_val_loss:
            print(f"  > Validation Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Create parent dir if not exists (should handle by config but safe check)
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print("  > Early stopping triggered!")
                break
    
    # Update best epoch in history
    history['best_epoch'] = best_epoch
    
    # Save training history to JSON
    history_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    return history
