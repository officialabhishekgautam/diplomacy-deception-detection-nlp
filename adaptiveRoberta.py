import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import logging
from tqdm import tqdm
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Any, Optional
from transformers import DataCollatorForLanguageModeling


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=1994):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

set_seed()

# Constants derived from EDA
SEASONS = ["Spring", "Fall", "Winter"]
MID_GAME_YEARS = ["1904", "1905", "1906", "1907"]
COUNTRIES = ["England", "France", "Germany", "Italy", "Austria", "Russia", "Turkey"]
COUNTRY_PAIRS = [(s, r) for s in COUNTRIES for r in COUNTRIES if s != r]
MAX_SEQ_LENGTH = 512
HIGH_DECEPTION_SENDERS = ["Italy", "Russia", "France"]
HIGH_TRUST_SENDERS = ["England", "Germany"]
HIGH_DECEPTION_PAIRS = [("Italy", "Austria"), ("Russia", "Turkey")]
HIGH_TRUST_PAIRS = [("Germany", "England")]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class FeatureEngineering:
    def __init__(self):
        # Initialize mappings for sender and receiver deception rates
        self.sender_deception_rates = {
            "Italy": 0.39,
            "Russia": 0.34,
            "France": 0.32,
            "Turkey": 0.28,
            "Austria": 0.25,
            "Germany": 0.22,
            "England": 0.18
        }
        
        # Initialize mappings for pair deception rates
        self.pair_deception_rates = {
            ("Italy", "Austria"): 0.44,
            ("Russia", "Turkey"): 0.38,
            ("Germany", "England"): 0.28
        }
        
        # Game-level deception rates
        self.game_deception_rates = {
            1: 0.084,
            2: 0.061,
            3: 0.030,
            5: 0.086,
            6: 0.122,
            7: 0.072,
            8: 0.051,
            9: 0.066,
            10: 0.049
        }
        
        # Initialize scalers
        self.numeric_scaler = StandardScaler()
        self.fitted = False
    
    def extract_features(self, data: List[Dict], fit=False) -> pd.DataFrame:
        """
        Extract features from raw data.
        
        Args:
            data: List of dialogue objects
            fit: Whether to fit the scaler (only for training data)
            
        Returns:
            DataFrame with extracted features
        """
        all_features = []
        
        for dialogue in data:
            # Skip empty dialogues
            if not dialogue.get("messages", []):
                continue
                
            game_id = dialogue.get("game_id", -1)
            messages = dialogue.get("messages", [])
            speakers = dialogue.get("speakers", [""] * len(messages))
            receivers = dialogue.get("receivers", [""] * len(messages))
            sender_labels = dialogue.get("sender_labels", [False] * len(messages))
            game_scores = dialogue.get("game_score", ["0"] * len(messages))
            game_score_deltas = dialogue.get("game_score_delta", ["0"] * len(messages))
            abs_indices = dialogue.get("absolute_message_index", list(range(len(messages))))
            rel_indices = dialogue.get("relative_message_index", list(range(len(messages))))
            seasons = dialogue.get("seasons", [""] * len(messages))
            years = dialogue.get("years", [""] * len(messages))
            
            # Calculate dialogue-level features
            dialogue_length = len(messages)
            word_counts = [len(msg.split()) if msg else 0 for msg in messages]
            avg_words = sum(word_counts) / dialogue_length if dialogue_length > 0 else 0
            lie_ratio = sum(1 for label in sender_labels if label) / dialogue_length if dialogue_length > 0 else 0
            
            # Get context for each message (last 3 messages)
            contexts = []
            span = 5
            for i in range(len(messages)):
                start = max(0, i - span)
                context = " ".join(messages[start:i])
                contexts.append(context)
            
            for i, (message, speaker, receiver, label, score, delta, abs_idx, rel_idx, season, year, context) in enumerate(
                zip(messages, speakers, receivers, sender_labels, game_scores, game_score_deltas, 
                    abs_indices, rel_indices, seasons, years, contexts)
            ):
                # Skip messages with missing metadata
                if not speaker or not receiver:
                    continue
                    
                # Basic message features
                word_count = len(message.split()) if message else 0
                is_empty = 1 if word_count == 0 else 0
                is_very_long = 1 if word_count > 97 else 0  # 97 is max from EDA
                
                # Punctuation features
                question_count = message.count('?') if message else 0
                exclamation_count = message.count('!') if message else 0
                ellipsis_count = message.count('...') if message else 0
                
                # Convert game scores to integers
                try:
                    game_score = int(score) if score.isdigit() else 0
                    game_score_delta = int(delta) if delta.isdigit() else 0
                except (ValueError, AttributeError):
                    game_score = 0
                    game_score_delta = 0
                
                # Calculate power difference
                power_diff = game_score  # Simplification since we don't have receiver's score
                
                # Temporal features
                season_spring = 1 if season == "Spring" else 0
                season_fall = 1 if season == "Fall" else 0
                season_winter = 1 if season == "Winter" else 0
                mid_game = 1 if year in MID_GAME_YEARS else 0
                
                # Position features
                early_dialogue = 1 if rel_idx < dialogue_length * 0.33 else 0
                mid_dialogue = 1 if 0.33 <= rel_idx/dialogue_length < 0.66 else 0
                late_dialogue = 1 if rel_idx >= dialogue_length * 0.66 else 0
                
                # Sender and receiver profiling
                sender_deception_rate = self.sender_deception_rates.get(speaker, 0.25)  # Default to average
                receiver_trust_rate = 1 - self.sender_deception_rates.get(receiver, 0.25)  # Inverse of deception rate
                
                # Pair-wise deception rate
                pair_deception_rate = self.pair_deception_rates.get((speaker, receiver), 0.25)  # Default to average
                
                # Game-level deception rate
                game_deception_rate = self.game_deception_rates.get(game_id, 0.06)  # Default to average
                
                # Indicators for high deception senders/pairs
                is_high_deception_sender = 1 if speaker in HIGH_DECEPTION_SENDERS else 0
                is_high_trust_sender = 1 if speaker in HIGH_TRUST_SENDERS else 0
                
                # Country one-hot encoding
                country_onehot = {c: 1 if speaker == c else 0 for c in COUNTRIES}
                
                # Combine all features
                features = {
                    "message": message,
                    "context": context,
                    "label": label,
                    "game_id": game_id,
                    "speaker": speaker,
                    "receiver": receiver,
                    "word_count": word_count,
                    "is_empty": is_empty,
                    "is_very_long": is_very_long,
                    "question_count": question_count,
                    "exclamation_count": exclamation_count,
                    "ellipsis_count": ellipsis_count,
                    "game_score": game_score,
                    "game_score_delta": game_score_delta,
                    "power_diff": power_diff,
                    "season_spring": season_spring,
                    "season_fall": season_fall,
                    "season_winter": season_winter,
                    "mid_game": mid_game,
                    "abs_message_index": abs_idx,
                    "rel_message_index": rel_idx,
                    "dialogue_length": dialogue_length,
                    "avg_dialogue_words": avg_words,
                    "dialogue_lie_ratio": lie_ratio,
                    "early_dialogue": early_dialogue,
                    "mid_dialogue": mid_dialogue,
                    "late_dialogue": late_dialogue,
                    "sender_deception_rate": sender_deception_rate,
                    "receiver_trust_rate": receiver_trust_rate,
                    "pair_deception_rate": pair_deception_rate,
                    "game_deception_rate": game_deception_rate,
                    "is_high_deception_sender": is_high_deception_sender,
                    "is_high_trust_sender": is_high_trust_sender,
                }
                
                # Add country one-hot features
                features.update({f"speaker_{country}": v for country, v in country_onehot.items()})
                
                all_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Define numeric columns for scaling
        numeric_cols = [
            "word_count", "game_score", "game_score_delta", "power_diff",
            "abs_message_index", "rel_message_index", "dialogue_length",
            "avg_dialogue_words", "dialogue_lie_ratio", "sender_deception_rate",
            "receiver_trust_rate", "pair_deception_rate", "game_deception_rate"
        ]
        
        # Scale numeric features
        if fit:
            self.numeric_scaler.fit(df[numeric_cols])
            self.fitted = True
            
        if self.fitted:
            df[numeric_cols] = self.numeric_scaler.transform(df[numeric_cols])
        
        return df

# Custom dataset for Diplomacy messages
class DiplomacyDataset(Dataset):
    def __init__(self, features, tokenizer, max_seq_length=512):
        self.features = features
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = self.features.iloc[idx]
        message = item["message"]
        context = item["context"]
        label = item["label"]
        
        # Create input text with context
        input_text = message
        if context:
            input_text = f"{message} [SEP] {context}"
        
        # Tokenize the input
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Extract metadata features
        metadata_features = [
            item["word_count"], item["is_empty"], item["is_very_long"], 
            item["question_count"], item["exclamation_count"], item["ellipsis_count"],
            item["game_score"], item["game_score_delta"], item["power_diff"],
            item["season_spring"], item["season_fall"], item["season_winter"],
            item["mid_game"], item["abs_message_index"], item["rel_message_index"],
            item["dialogue_length"], item["avg_dialogue_words"], item["dialogue_lie_ratio"],
            item["early_dialogue"], item["mid_dialogue"], item["late_dialogue"],
            item["sender_deception_rate"], item["receiver_trust_rate"], item["pair_deception_rate"],
            item["game_deception_rate"], item["is_high_deception_sender"], item["is_high_trust_sender"]
        ]
        
        # Add country one-hot features
        for country in COUNTRIES:
            metadata_features.append(item.get(f"speaker_{country}", 0))
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "metadata": torch.tensor(metadata_features, dtype=torch.float),
            "labels": torch.tensor(1 if label else 0, dtype=torch.long)
        }

# Define focal loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Custom RoBERTa model with metadata integration
class MetadataRoBERTa(nn.Module):
    def __init__(self, pretrained_model_name, num_metadata_features, hidden_dropout_prob=0.3):
        super(MetadataRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.metadata_projection = nn.Linear(num_metadata_features, 128)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Classifier head
        self.classifier = nn.Linear(self.roberta.config.hidden_size + 128, 1)
        
    def forward(self, input_ids, attention_mask, metadata):
        # Get RoBERTa embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        # Project metadata features
        metadata_embedding = self.metadata_projection(metadata)
        
        # Combine RoBERTa embeddings with metadata
        combined_embedding = torch.cat([cls_output, metadata_embedding], dim=1)
        combined_embedding = self.dropout(combined_embedding)
        
        # Classification
        logits = self.classifier(combined_embedding)
        
        return logits.squeeze(-1)

# Domain-adaptive pre-training with MLM
def domain_adaptive_pretraining(tokenizer, raw_texts, output_dir, epochs=3): #3 before
    logger.info("Starting domain-adaptive pre-training")
    
    # Create a dataset for masked language modeling
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=512):
            self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
            
        def __len__(self):
            return len(self.encodings["input_ids"])
        
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    # Prepare dataset
    dataset = TextDataset(raw_texts, tokenizer, max_length=MAX_SEQ_LENGTH)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)
    
    # Load pre-trained model for masked language modeling
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    
    # Save the domain-adapted model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Domain-adapted model saved to {output_dir}")
    
    return output_dir

# Training function
def train_model(model, train_dataloader, val_dataloader, class_weight, epochs=3, patience=10):
    logger.info("Starting model training")
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Loss function with class weights for imbalance
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1s = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                metadata=batch["metadata"]
            )
            
            # Calculate loss with class weights
            if class_weight > 1:
                weights = torch.ones_like(batch["labels"], device=device, dtype=torch.float)
                weights[batch["labels"] == 1] = class_weight
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float(), weight=weights)
            else:
                loss = criterion(logits, batch["labels"])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    metadata=batch["metadata"]
                )
                
                # Calculate loss
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
                val_loss += loss.item()
                
                # Get predictions
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        val_f1s.append(f1)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Macro F1: {f1:.4f}")
        
        # Early stopping
        if f1 > best_val_f1:
            best_val_f1 = f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pt")
            logger.info("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curves.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(val_f1s, label="Validation Macro F1")
    plt.title("Validation Macro F1")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("f1_curve.png")
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    
    return model, best_val_f1

# Evaluation function
def evaluate_model(model, test_dataloader):
    logger.info("Evaluating model on test set")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                metadata=batch["metadata"]
            )
            
            # Get predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch["labels"].cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Truthful", "Deceptive"])
    plt.yticks([0, 1], ["Truthful", "Deceptive"])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test Macro F1: {f1:.4f}")
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "roc_auc": roc_auc
    }

# Main pipeline
def main():
    # 1. Load data
    logger.info("Loading dataset")
    try:
        with open("train.jsonl", "r") as f:
            train_data = [json.loads(line) for line in f]
        
        with open("validation.jsonl", "r") as f:
            val_data = [json.loads(line) for line in f]
            
        with open("test.jsonl", "r") as f:
            test_data = [json.loads(line) for line in f]
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # 2. Feature engineering
    logger.info("Extracting features")
    fe = FeatureEngineering()
    train_features = fe.extract_features(train_data, fit=True)
    val_features = fe.extract_features(val_data)
    test_features = fe.extract_features(test_data)
    
    # 3. Prepare for domain adaptation
    logger.info("Preparing for domain adaptation")
    all_messages = []
    for data in [train_data, val_data, test_data]:
        for dialogue in data:
            all_messages.extend(dialogue.get("messages", []))
    
    all_messages = [msg for msg in all_messages if msg]  # Remove empty messages
    
    # 4. Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # 5. Domain-adaptive pre-training
    domain_adapted_model_path = domain_adaptive_pretraining(
        tokenizer=tokenizer,
        raw_texts=all_messages,
        output_dir="diplomacy_adapted_roberta",
        epochs=3
    )
    
    # 6. Calculate class weights for imbalance
    pos_count = sum(train_features["label"])
    neg_count = len(train_features) - pos_count
    class_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Class imbalance - Positive: {pos_count}, Negative: {neg_count}, Weight: {class_weight:.2f}")
    
    # 7. Create datasets
    train_dataset = DiplomacyDataset(train_features, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    val_dataset = DiplomacyDataset(val_features, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    test_dataset = DiplomacyDataset(test_features, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    
    # 8. Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    
    # 9. Initialize model
    num_metadata_features = len(train_dataset[0]["metadata"])
    model = MetadataRoBERTa(
        pretrained_model_name=domain_adapted_model_path,
        num_metadata_features=num_metadata_features,
        hidden_dropout_prob=0.1
    )
    model.to(device)
    
    # 10. Train model
    trained_model, best_val_f1 = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        class_weight=class_weight,
        epochs=3, #3 before
        patience=5
    )
    
    # 11. Evaluate on test set
    test_metrics = evaluate_model(trained_model, test_dataloader)
    
    # 12. Log final results
    logger.info("\n" + "="*50)
    logger.info("FINAL RESULTS")
    logger.info("="*50)
    logger.info(f"Best Validation F1: {best_val_f1:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test Macro F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    logger.info("="*50)
    
    # 13. Save test predictions for analysis
    logger.info("Generating test predictions for analysis")
    test_predictions = []
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                metadata=batch["metadata"]
            )
            
            # Get predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch["labels"].cpu().numpy()
            
            # Get original features
            start_idx = i * test_dataloader.batch_size
            end_idx = min((i + 1) * test_dataloader.batch_size, len(test_features))
            batch_features = test_features.iloc[start_idx:end_idx]
            
            # Add predictions to features
            for j, (prob, pred, label) in enumerate(zip(probs, preds, labels)):
                if start_idx + j < len(test_features):
                    test_predictions.append({
                        "message": batch_features.iloc[j]["message"],
                        "speaker": batch_features.iloc[j]["speaker"],
                        "receiver": batch_features.iloc[j]["receiver"],
                        "game_id": batch_features.iloc[j]["game_id"],
                        "true_label": bool(label),
                        "predicted_label": bool(pred),
                        "prediction_probability": float(prob),
                        "correct": label == pred
                    })
    
    # Save predictions to file
    pd.DataFrame(test_predictions).to_csv("test_predictions.csv", index=False)
    logger.info("Test predictions saved to test_predictions.csv")
    
    # 14. Analyze predictions by different factors
    predictions_df = pd.DataFrame(test_predictions)
    
    # Analyze by speaker
    speaker_performance = predictions_df.groupby("speaker").agg({
        "correct": "mean",
        "true_label": "mean",
        "predicted_label": "mean",
        "prediction_probability": "mean"
    }).reset_index()
    
    logger.info("\nPerformance by speaker country:")
    logger.info(speaker_performance.to_string())
    
    # Analyze by game_id
    game_performance = predictions_df.groupby("game_id").agg({
        "correct": "mean",
        "true_label": "mean",
        "predicted_label": "mean",
        "prediction_probability": "mean"
    }).reset_index()
    
    logger.info("\nPerformance by game_id:")
    logger.info(game_performance.to_string())
    
    # 15. Plot error analysis
    # Error analysis by probability threshold
    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    for threshold in thresholds:
        preds = (predictions_df["prediction_probability"] > threshold).astype(int)
        labels = predictions_df["true_label"].astype(int)
        
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Performance Metrics by Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig("threshold_analysis.png")
    
    logger.info("\nAnalysis complete! Results saved to CSV files and plots.")
    
    return {
        "model": trained_model,
        "tokenizer": tokenizer,
        "feature_engineering": fe,
        "test_metrics": test_metrics,
        "best_val_f1": best_val_f1,
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
        "class_weight": class_weight
    }

# Feature importance analysis
def analyze_feature_importance(model, test_dataloader):
    """
    Analyze feature importance by measuring impact on predictions when features are zeroed out.
    """
    logger.info("Analyzing feature importance")
    
    # Get baseline predictions
    model.eval()
    baseline_preds = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                metadata=batch["metadata"]
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            baseline_preds.extend(probs)
    
    # Get metadata feature names
    metadata_features = [
        "word_count", "is_empty", "is_very_long", 
        "question_count", "exclamation_count", "ellipsis_count",
        "game_score", "game_score_delta", "power_diff",
        "season_spring", "season_fall", "season_winter",
        "mid_game", "abs_message_index", "rel_message_index",
        "dialogue_length", "avg_dialogue_words", "dialogue_lie_ratio",
        "early_dialogue", "mid_dialogue", "late_dialogue",
        "sender_deception_rate", "receiver_trust_rate", "pair_deception_rate",
        "game_deception_rate", "is_high_deception_sender", "is_high_trust_sender"
    ]
    
    # Add country features
    for country in COUNTRIES:
        metadata_features.append(f"speaker_{country}")
    
    # Analyze each feature
    feature_importance = {}
    
    for i, feature_name in enumerate(metadata_features):
        # Zero out the feature
        modified_preds = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Create a copy of metadata and zero out the feature
                modified_metadata = batch["metadata"].clone()
                modified_metadata[:, i] = 0
                
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    metadata=modified_metadata
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                modified_preds.extend(probs)
        
        # Calculate impact (mean absolute difference)
        impact = np.mean(np.abs(np.array(baseline_preds) - np.array(modified_preds)))
        feature_importance[feature_name] = impact
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    features = [f[0] for f in sorted_features[:15]]  # Top 15 features
    importances = [f[1] for f in sorted_features[:15]]
    
    plt.barh(features, importances)
    plt.xlabel("Feature Importance (Impact on Predictions)")
    plt.title("Top 15 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    logger.info("Feature importance analysis complete")
    logger.info("\nTop 10 most important features:")
    for i, (feature, importance) in enumerate(sorted_features[:10]):
        logger.info(f"{i+1}. {feature}: {importance:.4f}")
    
    return feature_importance

# Run ablation study to measure impact of different components
def run_ablation_study(train_features, val_features, test_features, tokenizer, domain_adapted_model_path, class_weight):
    """
    Run ablation study to measure the impact of different components on model performance.
    """
    logger.info("\n" + "="*50)
    logger.info("ABLATION STUDY")
    logger.info("="*50)
    
    # Define ablation configurations
    ablation_configs = {
        "Full Model": {
            "use_domain_adaptation": True,
            "use_metadata": True,
            "use_context": True
        },
        "No Domain Adaptation": {
            "use_domain_adaptation": False,
            "use_metadata": True,
            "use_context": True
        },
        "No Metadata": {
            "use_domain_adaptation": True,
            "use_metadata": False,
            "use_context": True
        },
        "No Context": {
            "use_domain_adaptation": True,
            "use_metadata": True,
            "use_context": False
        },
        "No Metadata & No Context": {
            "use_domain_adaptation": True,
            "use_metadata": False,
            "use_context": False
        },
        "Only Text": {
            "use_domain_adaptation": False,
            "use_metadata": False,
            "use_context": True
        }
    }
    
    ablation_results = {}
    
    for config_name, config in ablation_configs.items():
        logger.info(f"\nRunning ablation config: {config_name}")
        
        # Create datasets based on configuration
        class CustomDataset(Dataset):
            def __init__(self, features, tokenizer, max_seq_length=512, use_context=True, use_metadata=True):
                self.features = features
                self.tokenizer = tokenizer
                self.max_seq_length = max_seq_length
                self.use_context = use_context
                self.use_metadata = use_metadata
                
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                item = self.features.iloc[idx]
                message = item["message"]
                context = item["context"] if self.use_context else ""
                label = item["label"]
                
                # Create input text with context if enabled
                input_text = message
                if self.use_context and context:
                    input_text = f"{message} [SEP] {context}"
                
                # Tokenize the input
                encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                result = {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(1 if label else 0, dtype=torch.long)
                }
                
                # Add metadata if enabled
                if self.use_metadata:
                    metadata_features = [
                        item["word_count"], item["is_empty"], item["is_very_long"], 
                        item["question_count"], item["exclamation_count"], item["ellipsis_count"],
                        item["game_score"], item["game_score_delta"], item["power_diff"],
                        item["season_spring"], item["season_fall"], item["season_winter"],
                        item["mid_game"], item["abs_message_index"], item["rel_message_index"],
                        item["dialogue_length"], item["avg_dialogue_words"], item["dialogue_lie_ratio"],
                        item["early_dialogue"], item["mid_dialogue"], item["late_dialogue"],
                        item["sender_deception_rate"], item["receiver_trust_rate"], item["pair_deception_rate"],
                        item["game_deception_rate"], item["is_high_deception_sender"], item["is_high_trust_sender"]
                    ]
                    
                    for country in COUNTRIES:
                        metadata_features.append(item.get(f"speaker_{country}", 0))
                    
                    result["metadata"] = torch.tensor(metadata_features, dtype=torch.float)
                
                return result
        
        # Create datasets and dataloaders
        train_dataset = CustomDataset(
            train_features, tokenizer, 
            use_context=config["use_context"], 
            use_metadata=config["use_metadata"]
        )
        val_dataset = CustomDataset(
            val_features, tokenizer, 
            use_context=config["use_context"], 
            use_metadata=config["use_metadata"]
        )
        test_dataset = CustomDataset(
            test_features, tokenizer, 
            use_context=config["use_context"], 
            use_metadata=config["use_metadata"]
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8)
        test_dataloader = DataLoader(test_dataset, batch_size=8)
        
        # Initialize model based on configuration
        model_name = domain_adapted_model_path if config["use_domain_adaptation"] else "roberta-base"
        
        if config["use_metadata"]:
            num_metadata_features = len(train_dataset[0]["metadata"])
            model = MetadataRoBERTa(
                pretrained_model_name=model_name,
                num_metadata_features=num_metadata_features,
                hidden_dropout_prob=0.1
            )
        else:
            # Create simpler model without metadata
            class SimpleRoBERTa(nn.Module):
                def __init__(self, pretrained_model_name, hidden_dropout_prob=0):
                    super(SimpleRoBERTa, self).__init__()
                    self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
                    self.dropout = nn.Dropout(hidden_dropout_prob)
                    self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)
                    
                def forward(self, input_ids, attention_mask, metadata=None):
                    outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    cls_output = outputs.last_hidden_state[:, 0, :]
                    cls_output = self.dropout(cls_output)
                    logits = self.classifier(cls_output)
                    return logits.squeeze(-1)
            
            model = SimpleRoBERTa(pretrained_model_name=model_name, hidden_dropout_prob=0)
        
        model.to(device)
        
        # Train model with shortened training for ablation study
        trained_model, best_val_f1 = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            class_weight=class_weight,
            epochs=3,  # Reduced epochs for ablation
            patience=3   # Reduced patience for ablation
        )
        
        # Evaluate on test set
        test_metrics = evaluate_model(trained_model, test_dataloader)
        
        # Store results
        ablation_results[config_name] = {
            "val_f1": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"]
        }
    
    # Create comparison table
    results_df = pd.DataFrame(ablation_results).T
    logger.info("\nAblation Study Results:")
    logger.info("\n" + results_df.to_string())
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    results_df[["test_accuracy", "test_precision", "test_recall", "test_f1"]].plot(kind="bar", figsize=(12, 6))
    plt.title("Ablation Study Results")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("ablation_study.png")
    
    return ablation_results

# Run the pipeline with error handling
if __name__ == "__main__":
    try:
        pipeline_results = main()
        train_features = pipeline_results["train_features"]
        val_features   = pipeline_results["val_features"]
        test_features  = pipeline_results["test_features"]
        class_weight   = pipeline_results["class_weight"]

        
        # Run additional analyses
        if pipeline_results:
            # Analyze feature importance
            feature_importance = analyze_feature_importance(
                pipeline_results["model"],
                DataLoader(DiplomacyDataset(
                    test_features, pipeline_results["tokenizer"], max_seq_length=MAX_SEQ_LENGTH
                ), batch_size=8)
            )
            
            # Run ablation study
            ablation_results = run_ablation_study(
                train_features, val_features, test_features,
                pipeline_results["tokenizer"],
                domain_adapted_model_path="diplomacy_adapted_roberta",
                class_weight=class_weight
            )
            
            logger.info("\nPipeline completed successfully!")
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    