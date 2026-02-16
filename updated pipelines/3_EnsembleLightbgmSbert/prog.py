import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import time
import torch
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(1994)
torch.manual_seed(1994)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DiplomacyDeceptionDetector:
    def __init__(self, model_name='all-mpnet-base-v2', max_seq_length=128, pca_components=256, 
                 use_pca=False, random_state=1994):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.pca_components = pca_components
        self.use_pca = use_pca
        self.random_state = random_state
        self.sentence_model = None
        self.pca = None
        self.lgbm_text = None
        self.lgbm_meta = None
        self.lgbm_combined = None
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.country_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.tf_idf = None
        
        # Initialize SBERT model
        print("Loading SBERT model...")
        self.sentence_model = SentenceTransformer(model_name)
        self.sentence_model.to(device)
        print("SBERT model loaded")
        
    def load_data(self, file_path):
        """Load and parse JSONL file into a structured format"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def process_dialogue(self, dialogue):
        """Extract messages and their corresponding features from a dialogue"""
        messages = []
        labels = []
        features = []
        
        # Check if dialogue has required fields
        if not dialogue.get('messages') or not dialogue.get('sender_labels'):
            return messages, labels, features
        
        for i, (msg, label) in enumerate(zip(dialogue['messages'], dialogue['sender_labels'])):
            if msg is None:
                continue
                
            # Basic message features
            msg_dict = {
                'message': msg,
                'is_lie': label,
                'game_id': dialogue.get('game_id', -1),
                'word_count': len(msg.split()) if msg else 0,
                'char_count': len(msg) if msg else 0,
                'is_empty': len(msg) == 0 if msg else True,
                'is_very_long': len(msg.split()) > 97 if msg else False,
                'question_marks': msg.count('?') if msg else 0,
                'exclamation_marks': msg.count('!') if msg else 0,
                'ellipsis': msg.count('...') if msg else 0,
            }
            
            # Add position features
            if 'absolute_message_index' in dialogue and i < len(dialogue['absolute_message_index']):
                msg_dict['absolute_message_index'] = dialogue['absolute_message_index'][i]
            else:
                msg_dict['absolute_message_index'] = i
                
            if 'relative_message_index' in dialogue and i < len(dialogue['relative_message_index']):
                msg_dict['relative_message_index'] = dialogue['relative_message_index'][i]
            else:
                msg_dict['relative_message_index'] = i / max(len(dialogue['messages']), 1)
            
            # Add season and year features
            if 'seasons' in dialogue and i < len(dialogue['seasons']):
                msg_dict['season'] = dialogue['seasons'][i]
                msg_dict['is_fall'] = dialogue['seasons'][i] == 'Fall'
            else:
                msg_dict['season'] = 'Unknown'
                msg_dict['is_fall'] = False
                
            if 'years' in dialogue and i < len(dialogue['years']):
                year = dialogue['years'][i]
                msg_dict['year'] = year
                msg_dict['is_mid_game'] = '1904' <= year <= '1907'
            else:
                msg_dict['year'] = 'Unknown'
                msg_dict['is_mid_game'] = False
            
            # Add sender and receiver features
            if 'speakers' in dialogue and i < len(dialogue['speakers']):
                msg_dict['sender'] = dialogue['speakers'][i]
            else:
                msg_dict['sender'] = 'Unknown'
                
            if 'receivers' in dialogue and i < len(dialogue['receivers']):
                msg_dict['receiver'] = dialogue['receivers'][i]
            else:
                msg_dict['receiver'] = 'Unknown'
            
            # Add power score features
            if 'game_score' in dialogue and i < len(dialogue['game_score']):
                try:
                    msg_dict['game_score'] = int(dialogue['game_score'][i])
                except (ValueError, TypeError):
                    msg_dict['game_score'] = 0
            else:
                msg_dict['game_score'] = 0
                
            if 'game_score_delta' in dialogue and i < len(dialogue['game_score_delta']):
                try:
                    msg_dict['game_score_delta'] = int(dialogue['game_score_delta'][i])
                except (ValueError, TypeError):
                    msg_dict['game_score_delta'] = 0
            else:
                msg_dict['game_score_delta'] = 0
            
            # Contextual features - previous message content is added during feature engineering
            
            messages.append(msg)
            labels.append(0 if label else 1)   # 1 = lie, 0 = truth
            features.append(msg_dict)
        
        return messages, labels, features
    
    def process_data(self, data):
        """Process all dialogues in dataset"""
        all_messages = []
        all_labels = []
        all_features = []
        
        for dialogue in data:
            messages, labels, features = self.process_dialogue(dialogue)
            all_messages.extend(messages)
            all_labels.extend(labels)
            all_features.extend(features)
        
        return all_messages, all_labels, all_features
    
    def engineer_features(self, features_list, train_mode=True):
        """Engineer features from raw message data"""
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        if len(df) == 0:
            print("Warning: Empty feature list")
            return pd.DataFrame(), []
        
        # Create sender_receiver pair feature
        df['sender_receiver_pair'] = df['sender'] + '_' + df['receiver']
        
        # One-hot encode categorical features
        if train_mode:
            countries = np.unique(list(df['sender'].unique()) + list(df['receiver'].unique()))
            self.country_encoder.fit(countries.reshape(-1, 1))
        
        sender_encoded = self.country_encoder.transform(df['sender'].values.reshape(-1, 1))
        receiver_encoded = self.country_encoder.transform(df['receiver'].values.reshape(-1, 1))
        
        # Create DataFrames with proper column names
        sender_cols = [f'sender_{c}' for c in self.country_encoder.categories_[0]]
        receiver_cols = [f'receiver_{c}' for c in self.country_encoder.categories_[0]]
        
        sender_df = pd.DataFrame(sender_encoded, columns=sender_cols)
        receiver_df = pd.DataFrame(receiver_encoded, columns=receiver_cols)
        
        # Combine with original DataFrame
        df = pd.concat([df, sender_df, receiver_df], axis=1)
        
        # Add power difference feature
        df['power_diff'] = df['game_score']
        
        # Create power advantage buckets
        df['power_advantage'] = pd.cut(df['power_diff'], 
                                       bins=[-float('inf'), -2, 2, float('inf')],
                                       labels=['disadvantage', 'equal', 'advantage'])
        
        # One-hot encode seasons and create fall flag
        df['is_fall'] = df['season'] == 'Fall'
        
        # Create mid-game indicator
        df['is_mid_game'] = df['year'].apply(lambda y: y in ['1904', '1905', '1906', '1907'] 
                                            if isinstance(y, str) else False)
        
        # Calculate linguistic markers
        df['formality_score'] = df['message'].apply(self._calculate_formality_score)
        df['certainty_score'] = df['message'].apply(self._calculate_certainty_score)
        df['emotionality_score'] = df['message'].apply(self._calculate_emotionality_score)
        
        # Create TF-IDF features if in training mode
        if train_mode:
            self.tf_idf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), 
                                           stop_words='english', min_df=3)
            tfidf_matrix = self.tf_idf.fit_transform(df['message'].fillna(''))
        else:
            if self.tf_idf is None:
                print("Warning: TF-IDF vectorizer not fitted. Skipping TF-IDF features.")
                tfidf_matrix = None
            else:
                tfidf_matrix = self.tf_idf.transform(df['message'].fillna(''))
        
        # Select features for modeling
        feature_columns = [
            'word_count', 'char_count', 'is_empty', 'is_very_long',
            'question_marks', 'exclamation_marks', 'ellipsis',
            'absolute_message_index', 'relative_message_index',
            'is_fall', 'is_mid_game', 'game_score', 'game_score_delta',
            'power_diff', 'formality_score', 'certainty_score', 'emotionality_score'
        ]
        
        # Add one-hot encoded country features
        feature_columns.extend(sender_cols)
        feature_columns.extend(receiver_cols)
        
        # Create final feature matrix
        X = df[feature_columns].fillna(0).copy()
        
        # Scale features
        if train_mode:
            self.scaler.fit(X)
        
        X_scaled = self.scaler.transform(X)
        
        # Add TF-IDF features if available
        if tfidf_matrix is not None:
            tfidf_array = tfidf_matrix.toarray()
            X_combined = np.hstack([X_scaled, tfidf_array])
        else:
            X_combined = X_scaled
        
        return df, X_combined
    
    def _calculate_formality_score(self, text):
        """Calculate a simple formality score based on linguistic markers"""
        if not text or not isinstance(text, str):
            return 0
            
        formal_markers = ['please', 'would', 'could', 'sincerely', 'appreciate', 'thank']
        informal_markers = ['hey', 'yeah', 'cool', 'ok', 'lol', 'haha', 'btw']
        
        text_lower = text.lower()
        formal_count = sum(text_lower.count(marker) for marker in formal_markers)
        informal_count = sum(text_lower.count(marker) for marker in informal_markers)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0
            
        return (formal_count - informal_count) / max(total_words, 1)
    
    def _calculate_certainty_score(self, text):
        """Calculate certainty score based on linguistic markers"""
        if not text or not isinstance(text, str):
            return 0
            
        certainty_markers = ['definitely', 'certainly', 'absolutely', 'sure', 'know', 'will']
        uncertainty_markers = ['maybe', 'perhaps', 'possibly', 'might', 'could be', 'not sure']
        
        text_lower = text.lower()
        certainty_count = sum(text_lower.count(marker) for marker in certainty_markers)
        uncertainty_count = sum(text_lower.count(marker) for marker in uncertainty_markers)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0
            
        return (certainty_count - uncertainty_count) / max(total_words, 1)
    
    def _calculate_emotionality_score(self, text):
        """Calculate emotionality score based on linguistic markers"""
        if not text or not isinstance(text, str):
            return 0
            
        emotion_markers = ['!', '?', 'love', 'hate', 'excited', 'worried', 'happy', 'sad']
        
        text_lower = text.lower()
        emotion_count = sum(text_lower.count(marker) for marker in emotion_markers)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0
            
        return emotion_count / max(total_words, 1)
    
    def get_sentence_embeddings(self, messages):
        """Get sentence embeddings using SBERT"""
        # Replace None values with empty strings
        messages = [msg if msg else "" for msg in messages]
        
        # Use batching for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]
            # Move to GPU for faster processing
            with torch.no_grad():
                batch_embeddings = self.sentence_model.encode(
                    batch, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=device
                )
            embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(embeddings)
        
        # Apply PCA if requested
        if self.use_pca:
            if self.pca is None:
                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
                embeddings = self.pca.fit_transform(embeddings)
            else:
                embeddings = self.pca.transform(embeddings)
                
        return embeddings
    
    def train_models(self, X_text, X_meta, y, n_folds=5):
        """Train the ensemble of LightGBM models using cross-validation"""
        print("Training models...")
        
        # Calculate class weights
        neg_pos_ratio = np.sum(y == 0) / max(np.sum(y == 1), 1)
        
        # Create cross-validation folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # Store predictions for meta-learner
        meta_train_preds = np.zeros((len(y), 6))  # 3 models x 2 class probabilities
        
        # LightGBM parameters
        lgbm_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'scale_pos_weight': neg_pos_ratio,
            'min_child_samples': 20,
            'verbosity': -1,
            'random_state': self.random_state,
        }
        
        # To store trained models
        text_models = []
        meta_models = []
        combined_models = []
        
        fold_metrics = []
        
        # Train models for each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y)):
            print(f"Training fold {fold+1}/{n_folds}")
            
            # Split data
            X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
            X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Combine text and meta features
            X_combined_train = np.hstack([X_text_train, X_meta_train])
            X_combined_val = np.hstack([X_text_val, X_meta_val])
            
            # Create LightGBM datasets
            dtrain_text = lgb.Dataset(X_text_train, label=y_train)
            dval_text = lgb.Dataset(X_text_val, label=y_val, reference=dtrain_text)
            
            dtrain_meta = lgb.Dataset(X_meta_train, label=y_train)
            dval_meta = lgb.Dataset(X_meta_val, label=y_val, reference=dtrain_meta)
            
            dtrain_combined = lgb.Dataset(X_combined_train, label=y_train)
            dval_combined = lgb.Dataset(X_combined_val, label=y_val, reference=dtrain_combined)
            
            # Train text-only model
            model_text = lgb.train(
                lgbm_params,
                dtrain_text,
                num_boost_round=100,
                valid_sets=[dtrain_text, dval_text],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            text_models.append(model_text)
            
            # Train metadata-only model
            model_meta = lgb.train(
                lgbm_params,
                dtrain_meta,
                num_boost_round=100,
                valid_sets=[dtrain_meta, dval_meta],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            meta_models.append(model_meta)
            
            # Train combined model
            model_combined = lgb.train(
                lgbm_params,
                dtrain_combined,
                num_boost_round=100,
                valid_sets=[dtrain_combined, dval_combined],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            combined_models.append(model_combined)
            
            # Generate predictions for meta-learner
            text_preds = model_text.predict(X_text_val)
            meta_preds = model_meta.predict(X_meta_val)
            comb_preds = model_combined.predict(X_combined_val)
            
            meta_train_preds[val_idx, 0] = 1 - text_preds
            meta_train_preds[val_idx, 1] = text_preds
            meta_train_preds[val_idx, 2] = 1 - meta_preds
            meta_train_preds[val_idx, 3] = meta_preds
            meta_train_preds[val_idx, 4] = 1 - comb_preds
            meta_train_preds[val_idx, 5] = comb_preds
            
            # Calculate metrics
            val_metrics = {}
            
            # Evaluate text model
            y_pred_text = np.round(text_preds).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_text, average='macro')
            acc = accuracy_score(y_val, y_pred_text)
            val_metrics['text'] = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
            
            # Evaluate meta model
            y_pred_meta = np.round(meta_preds).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_meta, average='macro')
            acc = accuracy_score(y_val, y_pred_meta)
            val_metrics['meta'] = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
            
            # Evaluate combined model
            y_pred_comb = np.round(comb_preds).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_comb, average='macro')
            acc = accuracy_score(y_val, y_pred_comb)
            val_metrics['combined'] = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
            
            fold_metrics.append(val_metrics)
            print(f"Fold {fold+1} Text F1: {val_metrics['text']['f1']:.4f}, "
                  f"Meta F1: {val_metrics['meta']['f1']:.4f}, "
                  f"Combined F1: {val_metrics['combined']['f1']:.4f}")
        
        # Average metrics across folds
        avg_metrics = {'text': {}, 'meta': {}, 'combined': {}}
        for model_type in ['text', 'meta', 'combined']:
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                avg_metrics[model_type][metric] = np.mean([m[model_type][metric] for m in fold_metrics])
        
        print("\nAverage Cross-Validation Metrics:")
        print(f"Text Model: Acc={avg_metrics['text']['accuracy']:.4f}, F1={avg_metrics['text']['f1']:.4f}")
        print(f"Meta Model: Acc={avg_metrics['meta']['accuracy']:.4f}, F1={avg_metrics['meta']['f1']:.4f}")
        print(f"Combined Model: Acc={avg_metrics['combined']['accuracy']:.4f}, F1={avg_metrics['combined']['f1']:.4f}")
        
        # Train meta-learner
        self.meta_learner = LogisticRegression(
            C=1.0, 
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state
        )
        self.meta_learner.fit(meta_train_preds, y)
        
        # Create final models on full dataset
        dtrain_text_full = lgb.Dataset(X_text, label=y)
        dtrain_meta_full = lgb.Dataset(X_meta, label=y)
        dtrain_combined_full = lgb.Dataset(np.hstack([X_text, X_meta]), label=y)
        
        self.lgbm_text = lgb.train(lgbm_params, dtrain_text_full, num_boost_round=100)
        self.lgbm_meta = lgb.train(lgbm_params, dtrain_meta_full, num_boost_round=100)
        self.lgbm_combined = lgb.train(lgbm_params, dtrain_combined_full, num_boost_round=100)
        
        return avg_metrics, fold_metrics
    
    def predict(self, X_text, X_meta):
        """Generate predictions using the trained ensemble"""
        if self.lgbm_text is None or self.lgbm_meta is None or self.lgbm_combined is None:
            raise ValueError("Models not trained. Call train_models first.")
        
        # Generate predictions from base models
        text_preds = self.lgbm_text.predict(X_text)
        meta_preds = self.lgbm_meta.predict(X_meta)
        X_combined = np.hstack([X_text, X_meta])
        comb_preds = self.lgbm_combined.predict(X_combined)
        
        # Prepare inputs for meta-learner
        meta_features = np.zeros((len(X_text), 6))
        meta_features[:, 0] = 1 - text_preds
        meta_features[:, 1] = text_preds
        meta_features[:, 2] = 1 - meta_preds
        meta_features[:, 3] = meta_preds
        meta_features[:, 4] = 1 - comb_preds
        meta_features[:, 5] = comb_preds
        
        # Get final predictions
        ensemble_probs = self.meta_learner.predict_proba(meta_features)[:, 1]
        ensemble_preds = np.round(ensemble_probs).astype(int)
        
        return {
            'text_probs': text_preds,
            'meta_probs': meta_preds,
            'combined_probs': comb_preds,
            'ensemble_probs': ensemble_probs,
            'predictions': ensemble_preds
        }
    
    def evaluate(self, y_true, pred_dict):
        """Evaluate predictions against true labels"""
        results = {}
        
        for model_name, probs in [
            ('text', pred_dict['text_probs']),
            ('meta', pred_dict['meta_probs']),
            ('combined', pred_dict['combined_probs']),
            ('ensemble', pred_dict['ensemble_probs'])
        ]:
            preds = np.round(probs).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average='macro', zero_division=0
            )
            
            # Calculate class-specific metrics
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                y_true, preds, average=None, zero_division=0
            )
            
            results[model_name] = {
                'accuracy': accuracy,
                'macro_precision': precision,
                'macro_recall': recall,
                'macro_f1': f1,
                'truth_precision': class_precision[0],
                'truth_recall': class_recall[0],
                'truth_f1': class_f1[0],
                'lie_precision': class_precision[1],
                'lie_recall': class_recall[1],
                'lie_f1': class_f1[1]
            }
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, preds)
            results[model_name]['confusion_matrix'] = cm
            
            # Calculate AUC
            fpr, tpr, _ = roc_curve(y_true, probs)
            results[model_name]['auc'] = auc(fpr, tpr)
            results[model_name]['fpr'] = fpr
            results[model_name]['tpr'] = tpr
        
        return results
    
    def plot_results(self, eval_results):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot model performance comparison
        model_names = list(eval_results.keys())
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        metric_values = {
            metric: [eval_results[model][metric] for model in model_names]
            for metric in metrics
        }
        
        ax = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width - 0.3, metric_values[metric], width, label=metric)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        # Plot confusion matrix for ensemble model
        cm = eval_results['ensemble']['confusion_matrix']
        ax = axes[0, 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix (Ensemble Model)')
        ax.set_xticklabels(['Truth', 'Lie'])
        ax.set_yticklabels(['Truth', 'Lie'])
        
        # Plot ROC curves
        ax = axes[1, 0]
        for model_name in model_names:
            results = eval_results[model_name]
            ax.plot(results['fpr'], results['tpr'], 
                   label=f'{model_name} (AUC = {results["auc"]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        
        # Plot class-specific F1 scores
        ax = axes[1, 1]
        truth_f1 = [eval_results[model]['truth_f1'] for model in model_names]
        lie_f1 = [eval_results[model]['lie_f1'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, truth_f1, width, label='Truth F1')
        ax.bar(x + width/2, lie_f1, width, label='Lie F1')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('F1 Score')
        ax.set_title('Class-specific F1 Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance for the LightGBM models"""
        if self.lgbm_meta is None:
            print("Models not trained. Call train_models first.")
            return
            
        # Get feature importance from metadata model
        meta_importance = self.lgbm_meta.feature_importance(importance_type='gain')
        
        # Ensure feature_names matches importance length
        if feature_names is None or len(feature_names) != len(meta_importance):
            feature_names = [f'feature_{i}' for i in range(len(meta_importance))]

        sorted_idx = np.argsort(meta_importance)

        top_n = 20  # Display top 20 features
        
        # Get top features
        top_features = [feature_names[i] for i in sorted_idx[-top_n:]]
        top_importance = meta_importance[sorted_idx[-top_n:]]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_importance)), top_importance, align='center')
        plt.yticks(range(len(top_importance)), top_features)
        plt.title('Feature Importance (Metadata Model)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self, train_path, val_path, test_path=None):
        """Run the full deception detection pipeline"""
        start_time = time.time()
        
        print("Loading data...")
        # Load training data
        train_data = self.load_data(train_path)
        train_messages, train_labels, train_features = self.process_data(train_data)
        
        # Load validation data
        val_data = self.load_data(val_path)
        val_messages, val_labels, val_features = self.process_data(val_data)
        
        print(f"Training data: {len(train_messages)} messages, {sum(train_labels)} deceptive")
        print(f"Validation data: {len(val_messages)} messages, {sum(val_labels)} deceptive")
        
        # Engineer features
        print("Engineering features...")
        train_df, train_meta_features = self.engineer_features(train_features, train_mode=True)
        val_df, val_meta_features = self.engineer_features(val_features, train_mode=False)
        
        # Get sentence embeddings
        print("Generating sentence embeddings...")
        train_text_features = self.get_sentence_embeddings(train_messages)
        val_text_features = self.get_sentence_embeddings(val_messages)
        
        # Train models
        print("Training models...")
        train_metrics, fold_metrics = self.train_models(
            train_text_features, train_meta_features, np.array(train_labels)
        )

        
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_predictions = self.predict(val_text_features, val_meta_features)
        val_results = self.evaluate(val_labels, val_predictions)
        
        # Print validation results
        print("\nValidation Results:")
        for model_name, results in val_results.items():
            print(f"{model_name.capitalize()} Model:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Macro F1: {results['macro_f1']:.4f}")
            print(f"  Lie F1: {results['lie_f1']:.4f}")
            print(f"  Truth F1: {results['truth_f1']:.4f}")
        
        # If test path provided, evaluate on test set
        test_results = None
        if test_path:
            print("\nEvaluating on test set...")
            test_data = self.load_data(test_path)
            test_messages, test_labels, test_features = self.process_data(test_data)
            
            test_df, test_meta_features = self.engineer_features(test_features, train_mode=False)
            test_text_features = self.get_sentence_embeddings(test_messages)
            
            test_predictions = self.predict(test_text_features, test_meta_features)
            test_results = self.evaluate(test_labels, test_predictions)
            
            print("\nTest Results:")
            for model_name, results in test_results.items():
                print(f"{model_name.capitalize()} Model:")
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  Macro F1: {results['macro_f1']:.4f}")
                print(f"  Lie F1: {results['lie_f1']:.4f}")
                print(f"  Truth F1: {results['truth_f1']:.4f}")
        
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        
        # Plot results
        self.plot_results(val_results)
        if test_results:
            self.plot_results(test_results)
        
        # Plot feature importance
        feature_names = list(train_df.columns)
        self.plot_feature_importance(feature_names)
        
        return {
            'train_metrics': train_metrics,
            'val_results': val_results,
            'test_results': test_results,
            'execution_time': end_time - start_time
        }


def main():
    # Set paths
    train_path = 'train.jsonl'
    val_path = 'validation.jsonl'
    test_path = 'test.jsonl'
    
    # Check if files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
    
    # Initialize detector
    detector = DiplomacyDeceptionDetector(
        model_name='all-mpnet-base-v2',
        max_seq_length=128,
        use_pca=False  # Set to True to reduce dimensionality
    )
    
    # Run pipeline
    results = detector.run_pipeline(train_path, val_path, test_path)
    
    print("\nPipeline complete! Results saved.")


if __name__ == "__main__":
    main()