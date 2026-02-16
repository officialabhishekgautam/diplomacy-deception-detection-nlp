import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import BertTokenizer, BertModel, AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import os
from collections import defaultdict, Counter
from torch.utils.data import WeightedRandomSampler
from torchcrf import CRF

# Set seed for reproducibility
SEED = 1994
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DiplomacyDataset:
    def __init__(self, file_path):
        """
        Load and preprocess data from a JSONL file.
        """
        self.data = []
        self.game_data = {}
        self.country_deception_rates = {}
        self.country_pair_deception_rates = {}
        
        # Load raw data
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Calculate deception rates for countries and country pairs
        self._calculate_deception_rates()
        
        # Process data
        self.processed_data = self._preprocess_data()
        
    def _calculate_deception_rates(self):
        """
        Calculate deception rates by country and country pairs from training data.
        """
        country_lies = defaultdict(int)
        country_messages = defaultdict(int)
        country_pair_lies = defaultdict(int)
        country_pair_messages = defaultdict(int)
        
        for dialogue in self.data:
            if 'messages' not in dialogue or not dialogue['messages']:
                continue
                
            game_id = dialogue.get('game_id', 'unknown')
            if game_id not in self.game_data:
                self.game_data[game_id] = {'lie_ratio': 0, 'total_messages': 0, 'deceptive_messages': 0}
            
            for i, (message, label) in enumerate(zip(dialogue['messages'], dialogue.get('sender_labels', []))):
                if not message.strip():  # Skip empty messages
                    continue
                    
                speaker = dialogue.get('speakers', [])[i] if i < len(dialogue.get('speakers', [])) else 'unknown'
                receiver = dialogue.get('receivers', [])[i] if i < len(dialogue.get('receivers', [])) else 'unknown'
                
                # Update country stats
                country_messages[speaker] += 1
                if label:  # If deceptive
                    country_lies[speaker] += 1
                
                # Update country pair stats
                pair_key = f"{speaker}→{receiver}"
                country_pair_messages[pair_key] += 1
                if label:  # If deceptive
                    country_pair_lies[pair_key] += 1
                    
                # Update game stats
                self.game_data[game_id]['total_messages'] += 1
                if label:
                    self.game_data[game_id]['deceptive_messages'] += 1
        
        # Calculate rates
        for country, messages in country_messages.items():
            self.country_deception_rates[country] = country_lies[country] / messages if messages > 0 else 0
        
        for pair, messages in country_pair_messages.items():
            self.country_pair_deception_rates[pair] = country_pair_lies[pair] / messages if messages > 0 else 0
            
        # Calculate game lie ratios
        for game_id in self.game_data:
            total = self.game_data[game_id]['total_messages']
            deceptive = self.game_data[game_id]['deceptive_messages']
            self.game_data[game_id]['lie_ratio'] = deceptive / total if total > 0 else 0
    
    def _preprocess_data(self):
        """
        Convert the raw JSON data into processed samples with features.
        """
        processed_samples = []
        
        for dialogue in self.data:
            if 'messages' not in dialogue or not dialogue['messages']:
                continue
                
            game_id = dialogue.get('game_id', 'unknown')
            game_lie_ratio = self.game_data[game_id]['lie_ratio'] if game_id in self.game_data else 0
            
            # Calculate dialogue-level features
            num_messages = len(dialogue['messages'])
            word_counts = [len(msg.split()) for msg in dialogue['messages']]
            avg_words = np.mean(word_counts) if word_counts else 0
            std_words = np.std(word_counts) if len(word_counts) > 1 else 0
            
            # Count truthful and deceptive messages in dialogue
            sender_labels = dialogue.get('sender_labels', [])
            true_count = sum(1 for label in sender_labels if not label)
            lie_count = sum(1 for label in sender_labels if label)
            lie_ratio = lie_count / num_messages if num_messages > 0 else 0
            
            # Process each message
            for i, message in enumerate(dialogue['messages']):
                if i >= len(dialogue.get('sender_labels', [])):
                    continue  # Skip if no label
                
                # Basic message features
                word_count = len(message.split())
                char_count = len(message)
                is_empty = word_count == 0
                is_very_long = word_count > 97  # Based on EDA max value
                
                # Punctuation counts
                question_count = message.count('?')
                exclamation_count = message.count('!')
                
                # Temporal features
                season = dialogue.get('seasons', [])[i] if i < len(dialogue.get('seasons', [])) else 'unknown'
                year = dialogue.get('years', [])[i] if i < len(dialogue.get('years', [])) else 'unknown'
                is_fall = season == 'Fall'  # EDA: Fall has higher deception rate
                
                try:
                    year_int = int(year) if year != 'unknown' else 0
                    is_mid_game = 1904 <= year_int <= 1907  # EDA: higher deception in mid-game
                except ValueError:
                    is_mid_game = False
                
                # Position in dialogue
                abs_msg_idx = dialogue.get('absolute_message_index', [])[i] if i < len(dialogue.get('absolute_message_index', [])) else 0
                rel_msg_idx = dialogue.get('relative_message_index', [])[i] if i < len(dialogue.get('relative_message_index', [])) else 0
                
                position_ratio = i / num_messages if num_messages > 0 else 0
                dialogue_phase = 0 if position_ratio < 0.33 else 1 if position_ratio < 0.66 else 2  # early, mid, late
                
                # Power features
                game_score = dialogue.get('game_score', [])[i] if i < len(dialogue.get('game_score', [])) else "0"
                game_score_delta = dialogue.get('game_score_delta', [])[i] if i < len(dialogue.get('game_score_delta', [])) else "0"
                
                try:
                    game_score = int(game_score)
                    game_score_delta = int(game_score_delta)
                except (ValueError, TypeError):
                    game_score = 0
                    game_score_delta = 0
                
                # Sender/Receiver features
                speaker = dialogue.get('speakers', [])[i] if i < len(dialogue.get('speakers', [])) else 'unknown'
                receiver = dialogue.get('receivers', [])[i] if i < len(dialogue.get('receivers', [])) else 'unknown'
                
                sender_deception_rate = self.country_deception_rates.get(speaker, 0)
                pair_key = f"{speaker}→{receiver}"
                pair_deception_rate = self.country_pair_deception_rates.get(pair_key, 0)
                
                # Label
                label = dialogue.get('sender_labels', [])[i]
                
                # Store processed sample
                sample = {
                    'game_id': game_id,
                    'message': message,
                    'label': 1 if label else 0,
                    
                    # Message features
                    'word_count': word_count,
                    'char_count': char_count,
                    'is_empty': 1 if is_empty else 0,
                    'is_very_long': 1 if is_very_long else 0,
                    'question_count': question_count,
                    'exclamation_count': exclamation_count,
                    
                    # Temporal features
                    'season': season,
                    'year': year,
                    'is_fall': 1 if is_fall else 0,
                    'is_mid_game': 1 if is_mid_game else 0,
                    'abs_msg_idx': abs_msg_idx,
                    'rel_msg_idx': rel_msg_idx,
                    'dialogue_phase': dialogue_phase,
                    
                    # Power features
                    'game_score': game_score,
                    'game_score_delta': game_score_delta,
                    
                    # Sender/Receiver features
                    'speaker': speaker,
                    'receiver': receiver,
                    'sender_deception_rate': sender_deception_rate,
                    'pair_deception_rate': pair_deception_rate,
                    
                    # Dialogue-level features
                    'num_messages': num_messages,
                    'avg_words': avg_words,
                    'std_words': std_words,
                    'dialogue_lie_ratio': lie_ratio,
                    'game_lie_ratio': game_lie_ratio,
                    
                    # Additional context
                    'dialogue_index': i,
                    'dialogue_id': id(dialogue)  # Unique ID for grouping messages within dialogue
                }
                
                processed_samples.append(sample)
        
        return processed_samples
    
    def get_data_as_df(self):
        """
        Return processed data as a pandas DataFrame.
        """
        return pd.DataFrame(self.processed_data)
    
    def get_metadata_dim(self):
        """
        Return the dimension of the metadata features vector.
        """
        # Count categorical features that will be one-hot encoded
        seasons = ['Spring', 'Fall', 'Winter', 'unknown']
        dialogue_phases = [0, 1, 2]  # early, mid, late
        
        # Base features (numeric)
        base_features = ['word_count', 'char_count', 'is_empty', 'is_very_long',
                        'question_count', 'exclamation_count', 'is_fall',
                        'is_mid_game', 'abs_msg_idx', 'rel_msg_idx',
                        'game_score', 'game_score_delta', 'sender_deception_rate',
                        'pair_deception_rate', 'num_messages', 'avg_words',
                        'std_words', 'dialogue_lie_ratio', 'game_lie_ratio']
        
        # Total dimension
        total_dim = len(base_features) + len(seasons) + len(dialogue_phases)
        return total_dim
    
    def get_labels(self):
        """
        Return all labels.
        """
        return [sample['label'] for sample in self.processed_data]
    
    def get_class_weights(self):
        """
        Calculate class weights based on class distribution with smoothing.
        """
        labels = self.get_labels()
        class_counts = Counter(labels)
        total = len(labels)
    
        # Apply square root to smooth the weights (less extreme)
        weights = {
            0: np.sqrt(total / (2 * class_counts[0])), 
            1: np.sqrt(total / (2 * class_counts[1]))
        }
        return weights



class DiplomacyBertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # One-hot encode season
        self.df['season_Spring'] = (self.df['season'] == 'Spring').astype(int)
        self.df['season_Fall'] = (self.df['season'] == 'Fall').astype(int)
        self.df['season_Winter'] = (self.df['season'] == 'Winter').astype(int)
        self.df['season_unknown'] = (self.df['season'] == 'unknown').astype(int)
        
        # One-hot encode dialogue phase
        self.df['phase_early'] = (self.df['dialogue_phase'] == 0).astype(int)
        self.df['phase_mid'] = (self.df['dialogue_phase'] == 1).astype(int)
        self.df['phase_late'] = (self.df['dialogue_phase'] == 2).astype(int)
        
        # Metadata features to use
        self.meta_features = [
            'word_count', 'char_count', 'is_empty', 'is_very_long',
            'question_count', 'exclamation_count', 'is_fall', 'is_mid_game',
            'abs_msg_idx', 'rel_msg_idx', 'game_score', 'game_score_delta',
            'sender_deception_rate', 'pair_deception_rate', 'num_messages',
            'avg_words', 'std_words', 'dialogue_lie_ratio', 'game_lie_ratio',
            'season_Spring', 'season_Fall', 'season_Winter', 'season_unknown',
            'phase_early', 'phase_mid', 'phase_late'
        ]
        
        # Normalize numeric features
        numeric_features = [
            'word_count', 'char_count', 'question_count', 'exclamation_count',
            'abs_msg_idx', 'rel_msg_idx', 'game_score', 'game_score_delta',
            'sender_deception_rate', 'pair_deception_rate', 'num_messages',
            'avg_words', 'std_words', 'dialogue_lie_ratio', 'game_lie_ratio'
        ]
        
        for feature in numeric_features:
            if self.df[feature].std() > 0:
                self.df[feature] = (self.df[feature] - self.df[feature].mean()) / self.df[feature].std()
            else:
                self.df[feature] = 0  # If std is 0, set all values to 0
    
    def __len__(self):
        return len(self.df)
        
    
    def __getitem__(self, idx):
        message = self.df.iloc[idx]['message']
        label = self.df.iloc[idx]['label']
        
        # Get metadata
        metadata = torch.tensor(self.df.iloc[idx][self.meta_features].astype(np.float32).values, dtype=torch.float)
        
        # Tokenize
        encoding = self.tokenizer(
            message,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': metadata,
            'label': torch.tensor(label, dtype=torch.long),
            'dialogue_id': self.df.iloc[idx]['game_id'],
            'dialogue_index': self.df.iloc[idx]['rel_msg_idx']
        }

def collate_dialogue_samples(batch):
    """
    Custom collate function for dialogue-level batching.
    Groups messages by dialogue_id and creates mini-batches.
    """
    # Group by dialogue_id
    dialogues = defaultdict(list)
    for item in batch:
        dialogue_id = item['dialogue_id']
        dialogues[dialogue_id].append(item)
    
    # Sort dialogues by length for better batching
    sorted_dialogues = sorted(dialogues.values(), key=len, reverse=True)
    
    # Create batches
    input_ids_batch = []
    attention_mask_batch = []
    metadata_batch = []
    label_batch = []
    dialogue_lens = []
    
    for dialogue in sorted_dialogues:
        # Sort messages by position in dialogue
        dialogue = sorted(dialogue, key=lambda x: x['dialogue_index'])
        
        # Extract tensors
        dialogue_input_ids = [msg['input_ids'] for msg in dialogue]
        dialogue_attention_mask = [msg['attention_mask'] for msg in dialogue]
        dialogue_metadata = [msg['metadata'] for msg in dialogue]
        dialogue_labels = [msg['label'] for msg in dialogue]
        
        # Pad to max dialogue length
        dialogue_len = len(dialogue)
        dialogue_lens.append(dialogue_len)
        
        # Stack tensors
        input_ids_batch.append(torch.stack(dialogue_input_ids))
        attention_mask_batch.append(torch.stack(dialogue_attention_mask))
        metadata_batch.append(torch.stack(dialogue_metadata))
        label_batch.append(torch.stack(dialogue_labels))
    
    # Pad to max dialogue length in batch
    max_dialogue_len = max(dialogue_lens)
    
    # Padding
    for i in range(len(input_ids_batch)):
        if input_ids_batch[i].size(0) < max_dialogue_len:
            padding_size = max_dialogue_len - input_ids_batch[i].size(0)
            
            # Create padding tensors
            input_ids_pad = torch.zeros(padding_size, input_ids_batch[i].size(1), dtype=input_ids_batch[i].dtype)
            attention_mask_pad = torch.zeros(padding_size, attention_mask_batch[i].size(1), dtype=attention_mask_batch[i].dtype)
            metadata_pad = torch.zeros(padding_size, metadata_batch[i].size(1), dtype=metadata_batch[i].dtype)
            label_pad = torch.zeros(padding_size, dtype=label_batch[i].dtype)
            
            # Pad tensors
            input_ids_batch[i] = torch.cat([input_ids_batch[i], input_ids_pad], dim=0)
            attention_mask_batch[i] = torch.cat([attention_mask_batch[i], attention_mask_pad], dim=0)
            metadata_batch[i] = torch.cat([metadata_batch[i], metadata_pad], dim=0)
            label_batch[i] = torch.cat([label_batch[i], label_pad], dim=0)
    
    return {
        'input_ids': torch.stack(input_ids_batch),
        'attention_mask': torch.stack(attention_mask_batch),
        'metadata': torch.stack(metadata_batch),
        'labels': torch.stack(label_batch),
        'dialogue_lens': torch.tensor(dialogue_lens)
    }

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, inputs, mask=None):
        # inputs: [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
        # mask: [batch_size, seq_len] or [seq_len]

        # Make sure inputs is 3D [batch_size, seq_len, hidden_size]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)  # Add batch dimension
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)  # Add batch dimension to mask too
        
        # Calculate attention scores
        scores = self.attention(inputs).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask
        if mask is not None:
            # Ensure mask is 2D
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        
        # Apply attention weights
        context = torch.bmm(weights.unsqueeze(1), inputs).squeeze(1)  # [batch_size, hidden]
        
        # Return same shape as input
        if inputs.size(0) == 1:
            return context.squeeze(0), weights.squeeze(0)
        else:
            return context, weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if weight is not None:
            weight = weight.float()  # 🔥 ensure float32
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        
    def forward(self, input, target):
        ce_loss = self.cross_entropy(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HierarchicalDeceptionModel(nn.Module):
    def __init__(self, bert_model_name, metadata_dim, hidden_size=128, dropout=0.5):

        dropout = 0.6

        super(HierarchicalDeceptionModel, self).__init__()
        self.hidden_size = hidden_size
        
        # BERT for token encoding
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # ── 3. UNFREEZE LAST 6 LAYERS ──
        # Freeze all BERT params
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 6 transformer layers
        for layer in self.bert.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
            
        # Message-level BiLSTM with more aggressive dropout
        self.message_lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if dropout > 0.3 else dropout,  # Add dropout in LSTM
            num_layers=2
        )
        
        # Message-level attention
        self.message_attention = AttentionLayer(hidden_size * 2)
        
        # Context-level BiLSTM
        self.context_lstm = nn.LSTM(
            hidden_size * 2,  # Output from message attention
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if dropout > 0.3 else dropout  # Add dropout in LSTM
        )
        
        # Context-level attention
        self.context_attention = AttentionLayer(hidden_size * 2)
        
        # Feature integration - Use LayerNorm
        self.feature_integration = nn.Sequential(
            nn.Linear(hidden_size * 2 + metadata_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Additional feature compression layer for regularization
        self.feature_compression = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            self.crf = CRF(num_tags=2, batch_first=True),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, input_ids, attention_mask, metadata, dialogue_lens):
        batch_size, max_dialogue_len, max_token_len = input_ids.size()
        
        # Process each message with BERT
        # Reshape for BERT processing
        input_ids_flat = input_ids.view(-1, max_token_len)
        attention_mask_flat = attention_mask.view(-1, max_token_len)
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_outputs = self.bert(
                input_ids=input_ids_flat,
                attention_mask=attention_mask_flat,
                return_dict=True
            )
        
        # Get sentence embeddings (CLS token)
        message_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size * max_dialogue_len, bert_hidden_size]
        
        # Reshape back to dialogue format
        message_embeddings = message_embeddings.view(batch_size, max_dialogue_len, -1)  # [batch_size, max_dialogue_len, bert_hidden_size]
        
        # Create dialogue mask from dialogue_lens
        dialogue_mask = torch.zeros(batch_size, max_dialogue_len, device=input_ids.device)
        for i, length in enumerate(dialogue_lens):
            dialogue_mask[i, :length] = 1
        
        # Process each dialogue with message-level LSTM
        message_lstm_output, _ = self.message_lstm(message_embeddings)  # [batch_size, max_dialogue_len, hidden_size*2]
        
        # Apply message-level attention to get context-aware message representations
        message_attn_outputs = []
        message_attentions = []
        
        for i in range(batch_size):
            valid_len = dialogue_lens[i]
            if valid_len > 0:
                msg_output, msg_attn = self.message_attention(
                    message_lstm_output[i, :valid_len, :],
                    dialogue_mask[i, :valid_len]
                )
                message_attn_outputs.append(msg_output)
                message_attentions.append(msg_attn)
            else:
                # Handle empty dialogues
                message_attn_outputs.append(torch.zeros(self.hidden_size * 2, device=input_ids.device))
                message_attentions.append(torch.zeros(0, device=input_ids.device))
        
        # Process dialogues with context-level LSTM
        context_outputs = []
        attentions = []
        
        for i in range(batch_size):
            valid_len = dialogue_lens[i]
            if valid_len > 0:
                # Get metadata for this dialogue
                dialogue_metadata = metadata[i, :valid_len, :]
                
                # Process with context LSTM
                context_input = message_lstm_output[i, :valid_len, :]
                context_output, _ = self.context_lstm(context_input.unsqueeze(0))
                context_output = context_output.squeeze(0)  # [valid_len, hidden_size*2]
                
                # Apply attention
                context_attn_mask = dialogue_mask[i, :valid_len]
                _, context_attn = self.context_attention(
                    context_output,
                    context_attn_mask
                )
                
                # Process each message in the dialogue
                for j in range(valid_len):

                    dialogue_metadata[j] = dialogue_metadata[j].float()  # Ensure float32

                    # Concatenate context output with metadata
                    combined_feature = torch.cat([context_output[j], dialogue_metadata[j]], dim=0).float()
                    
                    # Feature integration
                    integrated = self.feature_integration(combined_feature.unsqueeze(0))
                    
                    # Classification
                    logits = self.classifier(integrated.float())  # Force logits to float32
                    
                    context_outputs.append(logits)
                    attentions.append(context_attn[j] if context_attn.dim() > 0 else context_attn)
        
        if not context_outputs:
            return torch.zeros(0, 2, device=input_ids.device)
        
        # Stack outputs for all valid messages across all dialogues
        return torch.cat(context_outputs, dim=0)

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for batch in tqdm(train_loader, desc="Training"):

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['labels'].to(device)
        dialogue_lens = batch['dialogue_lens'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, metadata, dialogue_lens).float()

        
        # Flatten labels for loss calculation
        flat_labels = []
        for i, length in enumerate(dialogue_lens):
            flat_labels.extend(labels[i, :length].tolist())
        flat_labels = torch.tensor(flat_labels, dtype=torch.long).to(device)

        outputs = outputs.float()                # Model predictions → float32
        flat_labels = flat_labels.long()         # Labels → int64

        # Calculate loss
        loss = criterion(outputs, flat_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Save predictions and labels
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(flat_labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['labels'].to(device)
            dialogue_lens = batch['dialogue_lens'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, metadata, dialogue_lens).float()
            
            # Flatten labels for loss calculation
            flat_labels = []
            for i, length in enumerate(dialogue_lens):
                flat_labels.extend(labels[i, :length].tolist())
            flat_labels = torch.tensor(flat_labels, dtype=torch.long).to(device)
            
            if len(flat_labels) == 0 or len(outputs) == 0:
                continue  # Skip empty batches
                
            # Ensure types are correct
            outputs = outputs.float()                # Ensure logits are float32
            flat_labels = flat_labels.long()         # Ensure labels are int64

            # Calculate loss
            loss = criterion(outputs, flat_labels)
            
            total_loss += loss.item()
            
            # Save predictions and labels
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predictions.extend(preds)
            all_true_labels.extend(flat_labels.cpu().numpy())
    
    # Make sure we have predictions
    if len(all_predictions) == 0:
        print("Warning: No predictions were made during validation!")
        return {
            'loss': float('inf'),
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }, [], []
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    # Handle potential issues with precision_recall_fscore_support
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_predictions, average='macro', zero_division=0  # Changed variable names
        )
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_true_labels, all_predictions, zero_division=0  # Changed variable names
        )
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        precision, recall, f1 = 0, 0, 0
        class_precision = [0, 0]
        class_recall = [0, 0]
        class_f1 = [0, 0]
    
    metrics = {
        'loss': total_loss / max(1, len(val_loader)),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'truthful_precision': class_precision[0] if len(class_precision) > 0 else 0,
        'truthful_recall': class_recall[0] if len(class_recall) > 0 else 0,
        'truthful_f1': class_f1[0] if len(class_f1) > 0 else 0,
        'deceptive_precision': class_precision[1] if len(class_precision) > 1 else 0,
        'deceptive_recall': class_recall[1] if len(class_recall) > 1 else 0,
        'deceptive_f1': class_f1[1] if len(class_f1) > 1 else 0
    }
    
    return metrics, all_predictions, all_true_labels

def plot_metrics(train_metrics, val_metrics, metric_name, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name}_plot.png')
    plt.close()

def analyze_errors(predictions, true_labels, val_df):

    errors = np.array(predictions) != np.array(true_labels)
    error_indices = np.nonzero(errors)[0]
    
    if len(error_indices) == 0:
        print("No errors found in evaluation.")
        return {
            'total_errors': 0,
            'error_rate': 0,
            'false_positives': 0,
            'false_negatives': 0
        }, pd.DataFrame()

    # Extract error examples
    error_examples = val_df.iloc[error_indices].copy()
    error_examples['predicted'] = np.array(predictions)[error_indices]
    error_examples['true'] = np.array(true_labels)[error_indices]
    
    # Analyze false positives (predicted deceptive but actually truthful)
    fp = error_examples[error_examples['predicted'] == 1]
    
    # Analyze false negatives (predicted truthful but actually deceptive)
    fn = error_examples[error_examples['predicted'] == 0]
    
    # Return summary statistics
    error_summary = {
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(true_labels),
        'false_positives': len(fp),
        'false_negatives': len(fn)
    }
    
    # Calculate feature distributions for errors
    for feature in ['word_count', 'is_fall', 'is_mid_game', 'sender_deception_rate', 'game_lie_ratio']:
        if feature in error_examples.columns:
            error_summary[f'fp_{feature}_mean'] = fp[feature].mean() if len(fp) > 0 else 0
            error_summary[f'fn_{feature}_mean'] = fn[feature].mean() if len(fn) > 0 else 0
    
    print("\nError Analysis:")
    print(f"Total errors: {error_summary['total_errors']} ({error_summary['error_rate']*100:.2f}%)")
    print(f"False positives: {error_summary['false_positives']}")
    print(f"False negatives: {error_summary['false_negatives']}")
    
    return error_summary, error_examples

def augment_minority_class(df, minority_class=0, augmentation_factor=3):
    """
    Augment the minority class in the dataset by duplicating samples.
    
    Args:
        df: DataFrame containing the data
        minority_class: The class to augment (0 for truthful, 1 for deceptive)
        augmentation_factor: How many times to duplicate the minority class
        
    Returns:
        Augmented DataFrame
    """
    # Find minority samples
    minority_samples = df[df['label'] == minority_class]
    
    # Create augmented samples by duplicating minority samples
    augmented_samples = pd.concat([minority_samples] * augmentation_factor)
    
    # Combine with original data
    augmented_df = pd.concat([df, augmented_samples]).reset_index(drop=True)
    
    print(f"Augmented {len(minority_samples)} {minority_class} samples by factor of {augmentation_factor}")
    print(f"Original distribution: {len(df[df['label'] == 0])} truthful, {len(df[df['label'] == 1])} deceptive")
    print(f"Augmented distribution: {len(augmented_df[augmented_df['label'] == 0])} truthful, {len(augmented_df[augmented_df['label'] == 1])} deceptive")
    
    return augmented_df

def main():
    # Data paths
    train_path = 'data/train.jsonl'
    val_path = 'data/validation.jsonl'
    test_path = 'data/test.jsonl'
    
    # Load data
    print("Loading data...")
    train_data = DiplomacyDataset(train_path)
    val_data = DiplomacyDataset(val_path)
    test_data = DiplomacyDataset(test_path)
    
    # Convert to DataFrame
    train_df = train_data.get_data_as_df()
    val_df = val_data.get_data_as_df()
    test_df = test_data.get_data_as_df()
    
    # If you want to augment the minority class, uncomment the line below
    # Determine which class is minority first
    train_labels = train_df['label'].values
    minority_class = 0 if sum(train_labels) > (len(train_labels) - sum(train_labels)) else 1
    
    # Augment minority class if needed (uncomment to use)
    train_df = augment_minority_class(train_df, minority_class=minority_class, augmentation_factor=3)
    
    print(f"Train data: {len(train_df)} messages")
    print(f"Validation data: {len(val_df)} messages")
    print(f"Test data: {len(test_df)} messages")
    
    # Check class distribution
    train_labels = train_df['label'].values
    val_labels = val_df['label'].values
    test_labels = test_df['label'].values
    
    print("\nClass distribution:")
    print(f"Train: {len(train_labels) - sum(train_labels)} deceptive, {sum(train_labels)} truthful")
    print(f"Validation: {len(val_labels) - sum(val_labels)} deceptive, {sum(val_labels)} truthful")
    print(f"Test: {len(test_labels) - sum(test_labels)} deceptive, {sum(test_labels)} truthful")
    
    # Calculate class weights
    class_weights = train_data.get_class_weights()
    print(f"\nClass weights: {class_weights}")
    
    # Initialize tokenizer
    print("\nInitializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = DiplomacyBertDataset(train_df, tokenizer)
    val_dataset = DiplomacyBertDataset(val_df, tokenizer)
    test_dataset = DiplomacyBertDataset(test_df, tokenizer)
    
    # Create data loaders
    batch_size = 4  # Small batch size to handle dialogue-level batching
    
    # compute per-sample weights
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_dialogue_samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dialogue_samples
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dialogue_samples
    )
    
    # Initialize model
    print("\nInitializing model...")
    metadata_dim = train_data.get_metadata_dim()
    model = HierarchicalDeceptionModel('bert-base-uncased', metadata_dim).to(device)
    
    print(f"Model initialized with metadata dimension: {metadata_dim}")

    # Define num_epochs before calculating num_training_steps
    num_epochs = 8
    
    # Calculate training steps based on epochs and batch size
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(),           "lr": 3e-5},   # fine‑tune BERT gently
        {"params": list(model.message_lstm.parameters())
                   + list(model.context_lstm.parameters())
                   + list(model.feature_integration.parameters())
                   + list(model.classifier.parameters()),
         "lr": 1e-3},
    ], weight_decay=0.01)
    
    # Use class weights for weighted loss
    weight = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32, device=device)
    criterion = FocalLoss(weight=weight, gamma=2.0)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-3,
        total_steps=num_training_steps,
        pct_start=0.1,  # Warmup for first 10% of steps
        anneal_strategy='cos',
        div_factor=25.0
    )
    
    # Training loop
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        # Validate
        val_metrics, val_predictions, val_true_labels = validate(model, val_loader, criterion, device)
        
        # Store metrics for plotting
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_f1_scores.append(train_metrics['f1'])
        val_f1_scores.append(val_metrics['f1'])
        
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Deceptive - Precision: {val_metrics['truthful_precision']:.4f}, Recall: {val_metrics['truthful_recall']:.4f}, F1: {val_metrics['truthful_f1']:.4f}")
        print(f"Truthful - Precision: {val_metrics['deceptive_precision']:.4f}, Recall: {val_metrics['deceptive_recall']:.4f}, F1: {val_metrics['deceptive_f1']:.4f}")
        
        # Check for early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training metrics
    plot_metrics(train_losses, val_losses, 'Loss', 'Training and Validation Loss')
    plot_metrics(train_f1_scores, val_f1_scores, 'F1', 'Training and Validation F1')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Evaluate on validation set
    print("\nFinal evaluation on validation set:")
    val_metrics, val_predictions, val_true_labels = validate(model, val_loader, criterion, device)
    
    # Analyze validation errors
    error_summary, error_examples = analyze_errors(val_predictions, val_true_labels, val_df)
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_metrics, test_predictions, test_true_labels = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro Precision: {test_metrics['precision']:.4f}")
    print(f"Test Macro Recall: {test_metrics['recall']:.4f}")
    print(f"Test Macro F1: {test_metrics['f1']:.4f}")
    print(f"Test Deceptive - F1: {test_metrics['truthful_f1']:.4f}")
    print(f"Test Truthful - F1: {test_metrics['deceptive_f1']:.4f}")
    
    # Analyze test errors
    test_error_summary, test_error_examples = analyze_errors(test_predictions, test_true_labels, test_df)
    
    # Feature importance analysis
    print("\nMost important features for classifying deceptive messages:")
    if len(error_examples) > 0:
        deceptive_messages = train_df[train_df['label'] == 1]
        truthful_messages = train_df[train_df['label'] == 0]
        
        # Compare average values of key features
        features_to_analyze = ['word_count', 'char_count', 'question_count', 'exclamation_count', 
                              'is_fall', 'is_mid_game', 'sender_deception_rate', 'pair_deception_rate',
                              'dialogue_lie_ratio', 'game_lie_ratio']
        
        print("\nFeature analysis (mean values):")
        for feature in features_to_analyze:
            if feature in deceptive_messages.columns and feature in truthful_messages.columns:
                deceptive_mean = deceptive_messages[feature].mean()
                truthful_mean = truthful_messages[feature].mean()
                difference = deceptive_mean - truthful_mean
                print(f"{feature}: Deceptive={deceptive_mean:.4f}, Truthful={truthful_mean:.4f}, Diff={difference:.4f}")

if __name__ == "__main__":
    main()