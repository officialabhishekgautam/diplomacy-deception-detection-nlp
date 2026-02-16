import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import re
import warnings
import pickle
from collections import Counter
import datetime

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 1994
np.random.seed(RANDOM_SEED)

# Directory for saving outputs
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

class DiplomacyDeceptionDetector:
    def __init__(self):
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self.lightgbm_text = None
        self.lightgbm_meta = None
        self.lightgbm_combined = None
        self.meta_learner = None
        self.pca = None
        self.scaler = None
        self.encoder = None
        self.country_deception_rates = None
        self.pair_deception_rates = None
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load and preprocess data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    # Fix for the error in the extract_features method
    def extract_features(self, data_list, is_training=True):
        """Extract features from raw data"""
        all_features = []
        all_labels = []
    
        # Track country-level and pair-level deception rates
        if is_training:
            country_lie_counts = Counter()
            country_message_counts = Counter()
            pair_lie_counts = Counter()
            pair_message_counts = Counter()
    
        for game in tqdm(data_list, desc="Processing games"):
            if not game.get('messages') or len(game.get('messages', [])) == 0:
                continue
            
            game_id = game.get('game_id', 0)
            messages = game.get('messages', [])
            sender_labels = game.get('sender_labels', [])
            speakers = game.get('speakers', [])
            receivers = game.get('receivers', [])
            game_scores = game.get('game_score', [])
            game_score_deltas = game.get('game_score_delta', [])
            abs_msg_indices = game.get('absolute_message_index', [])
            rel_msg_indices = game.get('relative_message_index', [])
            seasons = game.get('seasons', [])
            years = game.get('years', [])
        
            # Calculate dialogue-level statistics
            num_messages = len(messages)
            words_per_message = [len(msg.split()) if msg else 0 for msg in messages]
            avg_words = np.mean(words_per_message) if words_per_message else 0
            lie_ratio = sum(1 for label in sender_labels if label) / len(sender_labels) if sender_labels else 0
        
            # Process each message in the dialogue
            for i in range(len(messages)):
                if i >= len(sender_labels):
                    continue
                
                # Get base data for this message
                message = messages[i] if i < len(messages) else ""
                label = sender_labels[i] if i < len(sender_labels) else False
                speaker = speakers[i] if i < len(speakers) else "unknown"
                receiver = receivers[i] if i < len(receivers) else "unknown"
            
                # Update country and pair stats if training
                if is_training:
                    country_message_counts[speaker] += 1
                    pair_key = f"{speaker}->{receiver}"
                    pair_message_counts[pair_key] += 1
                
                    if label:
                        country_lie_counts[speaker] += 1
                        pair_lie_counts[pair_key] += 1
            
                # Text features
                word_count = len(message.split()) if message else 0
                char_count = len(message) if message else 0
                is_empty = word_count == 0
                is_very_long = word_count > 97  # Based on EDA max
            
                # Punctuation features
                question_count = message.count('?') if message else 0
                exclamation_count = message.count('!') if message else 0
                ellipsis_count = message.count('...') if message else 0
            
                # Position features
                abs_index = abs_msg_indices[i] if i < len(abs_msg_indices) else 0
                rel_index = rel_msg_indices[i] if i < len(rel_msg_indices) else 0
                rel_position = rel_index / num_messages if num_messages > 0 else 0
            
                # Season & Year features
                season = seasons[i] if i < len(seasons) else "unknown"
                year = years[i] if i < len(years) else "unknown"
                is_fall = season == "Fall"
                try:
                    year_num = int(year) if year != "unknown" else 0
                    is_mid_game = 1904 <= year_num <= 1907
                except ValueError:
                    year_num = 0
                    is_mid_game = False
            
                # Game state features - FIX HERE for the type error
                # Check if game_scores[i] exists and handle different types
                sender_score = 0
                if i < len(game_scores):
                    try:
                        # If it's already an int
                        if isinstance(game_scores[i], int):
                           sender_score = game_scores[i]
                        # If it's a string that can be converted to int
                        elif isinstance(game_scores[i], str) and game_scores[i].isdigit():
                            sender_score = int(game_scores[i])
                    except:
                        sender_score = 0
            
                # Same fix for score_delta
                score_delta = 0
                if i < len(game_score_deltas):
                    try:
                        # If it's already an int
                        if isinstance(game_score_deltas[i], int):
                            score_delta = game_score_deltas[i]
                        # If it's a string that can be converted to int
                        elif isinstance(game_score_deltas[i], str) and game_score_deltas[i].isdigit():
                            score_delta = int(game_score_deltas[i])
                    except:
                        score_delta = 0
            
                # Context features - previous messages from the same dialogue
                context = messages[max(0, i-3):i]
                context_text = " ".join(context) if context else ""
            
                # Dialogue-level features
                features = {
                    # Message metadata
                    'game_id': game_id,
                    'speaker': speaker,
                    'receiver': receiver,
                    'season': season,
                    'year': year_num,
                
                    # Text features
                    'message': message,
                    'context': context_text,
                    'word_count': word_count,
                    'char_count': char_count,
                    'is_empty': is_empty,
                    'is_very_long': is_very_long,
                    'question_count': question_count,
                    'exclamation_count': exclamation_count,
                    'ellipsis_count': ellipsis_count,
                
                    # Position features
                    'absolute_index': abs_index,
                    'relative_index': rel_index,
                    'relative_position': rel_position,
                
                    # Season & Year features
                    'is_fall': is_fall,
                    'is_mid_game': is_mid_game,
                
                    # Game state features
                    'sender_score': sender_score,
                    'score_delta': score_delta,
                
                    # Dialogue-level statistics
                    'num_messages_in_dialogue': num_messages,
                    'avg_words_in_dialogue': avg_words,
                    'lie_ratio_in_dialogue': lie_ratio,
                }
            
                all_features.append(features)
                all_labels.append(label)
    
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        labels = np.array(all_labels)
    
        # Calculate and save country-level and pair-level deception rates if training
        if is_training:
            self.country_deception_rates = {country: country_lie_counts[country] / max(1, country_message_counts[country]) 
                                      for country in country_message_counts}
            self.pair_deception_rates = {pair: pair_lie_counts[pair] / max(1, pair_message_counts[pair]) 
                                   for pair in pair_message_counts}
        
            # Save these for later use
            with open(os.path.join(OUTPUT_DIR, "country_deception_rates.pkl"), 'wb') as f:
                pickle.dump(self.country_deception_rates, f)
            with open(os.path.join(OUTPUT_DIR, "pair_deception_rates.pkl"), 'wb') as f:
                pickle.dump(self.pair_deception_rates, f)
        else:
            # Load the pre-computed rates
            if self.country_deception_rates is None and os.path.exists(os.path.join(OUTPUT_DIR, "country_deception_rates.pkl")):
                with open(os.path.join(OUTPUT_DIR, "country_deception_rates.pkl"), 'rb') as f:
                    self.country_deception_rates = pickle.load(f)
            
            if self.pair_deception_rates is None and os.path.exists(os.path.join(OUTPUT_DIR, "pair_deception_rates.pkl")):
                with open(os.path.join(OUTPUT_DIR, "pair_deception_rates.pkl"), 'rb') as f:
                    self.pair_deception_rates = pickle.load(f)
    
        # Add country and pair deception rates to features
        if self.country_deception_rates:
            df['country_deception_rate'] = df['speaker'].map(lambda x: self.country_deception_rates.get(x, 0.05))
        else:
            df['country_deception_rate'] = 0.05  # Default
        
        if self.pair_deception_rates:
            df['pair_deception_rate'] = df.apply(lambda x: self.pair_deception_rates.get(f"{x['speaker']}->{x['receiver']}", 0.05), axis=1)
        else:
            df['pair_deception_rate'] = 0.05  # Default
    
        return df, labels

    def prepare_features(self, df, is_training=True):
        """Transform raw features into model-ready features"""
        # Text vectorization using TF-IDF
        if is_training:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
            tfidf_features = self.tfidf_vectorizer.fit_transform(df['message'].fillna(''))
            
            # Save vectorizer
            with open(os.path.join(OUTPUT_DIR, "models", "tfidf_vectorizer.pkl"), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        else:
            if self.tfidf_vectorizer is None:
                with open(os.path.join(OUTPUT_DIR, "models", "tfidf_vectorizer.pkl"), 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
            tfidf_features = self.tfidf_vectorizer.transform(df['message'].fillna(''))
        
        # Encode categorical features
        cat_features = ['speaker', 'receiver', 'season']
        if is_training:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_encoded = self.encoder.fit_transform(df[cat_features].fillna('unknown'))
            
            # Save encoder
            with open(os.path.join(OUTPUT_DIR, "models", "categorical_encoder.pkl"), 'wb') as f:
                pickle.dump(self.encoder, f)
        else:
            if self.encoder is None:
                with open(os.path.join(OUTPUT_DIR, "models", "categorical_encoder.pkl"), 'rb') as f:
                    self.encoder = pickle.load(f)
            cat_encoded = self.encoder.transform(df[cat_features].fillna('unknown'))
        
        # Numeric features to include
        numeric_features = [
            'word_count', 'char_count', 'is_empty', 'is_very_long',
            'question_count', 'exclamation_count', 'ellipsis_count',
            'absolute_index', 'relative_index', 'relative_position',
            'is_fall', 'is_mid_game', 'sender_score', 'score_delta',
            'num_messages_in_dialogue', 'avg_words_in_dialogue', 'lie_ratio_in_dialogue',
            'country_deception_rate', 'pair_deception_rate'
        ]
        
        # Scale numeric features
        if is_training:
            self.scaler = StandardScaler()
            numeric_scaled = self.scaler.fit_transform(df[numeric_features].fillna(0))
            
            # Save scaler
            with open(os.path.join(OUTPUT_DIR, "models", "numeric_scaler.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            if self.scaler is None:
                with open(os.path.join(OUTPUT_DIR, "models", "numeric_scaler.pkl"), 'rb') as f:
                    self.scaler = pickle.load(f)
            numeric_scaled = self.scaler.transform(df[numeric_features].fillna(0))
        
        # Generate sentence embeddings
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        print("Generating sentence embeddings...")
        embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df['message'].iloc[i:i+batch_size].fillna('').tolist()
            batch_embeddings = self.sentence_model.encode(batch)
            embeddings.extend(batch_embeddings)
        embeddings = np.array(embeddings)
        
        # Reduce dimensionality of embeddings if needed
        if is_training:
            self.pca = PCA(n_components=256)
            embeddings_reduced = self.pca.fit_transform(embeddings)
            
            # Save PCA
            with open(os.path.join(OUTPUT_DIR, "models", "pca_embeddings.pkl"), 'wb') as f:
                pickle.dump(self.pca, f)
        else:
            if self.pca is None:
                with open(os.path.join(OUTPUT_DIR, "models", "pca_embeddings.pkl"), 'rb') as f:
                    self.pca = pickle.load(f)
            embeddings_reduced = self.pca.transform(embeddings)
        
        # Context embeddings (previous messages)
        context_embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df['context'].iloc[i:i+batch_size].fillna('').tolist()
            batch_embeddings = self.sentence_model.encode(batch)
            context_embeddings.extend(batch_embeddings)
        context_embeddings = np.array(context_embeddings)
        
        # Reduce context embeddings dimensionality
        context_embeddings_reduced = self.pca.transform(context_embeddings)
        
        # Store feature names for later interpretation
        self.feature_names = {
            'numeric': numeric_features,
            'categorical': self.encoder.get_feature_names_out(cat_features),
            'tfidf': self.tfidf_vectorizer.get_feature_names_out()
        }
        
        # Prepare different feature sets
        X_text = embeddings_reduced
        X_context = context_embeddings_reduced
        X_meta = np.hstack([numeric_scaled, cat_encoded])
        X_tfidf = tfidf_features.toarray()
        
        # Combined features for the full model
        X_combined = np.hstack([X_meta, X_text, X_context])
        
        return {
            'text': X_text,
            'context': X_context, 
            'meta': X_meta,
            'tfidf': X_tfidf,
            'combined': X_combined
        }
    
    def train_models(self, X_features, y):
        """Train individual models and ensemble"""
        print("Training models...")
        
        # Calculate class weight to handle imbalance
        n_negative = sum(1 for label in y if not label)
        n_positive = sum(1 for label in y if label)
        scale_pos_weight = n_negative / max(1, n_positive)
        
        # Create cross-validation folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        
        # Store models and predictions for each fold
        self.models = {
            'text': [],
            'meta': [],
            'combined': []
        }
        
        cv_predictions = {
            'text': np.zeros(len(y)),
            'meta': np.zeros(len(y)),
            'combined': np.zeros(len(y))
        }
        
        # Train models with cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_features['combined'], y)):
            print(f"\nFold {fold+1}/5")
            
            # Text-only model (SBERT embeddings)
            print("Training text model...")
            X_text_train, X_text_val = X_features['text'][train_idx], X_features['text'][val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            text_model = lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED
            )
            
            text_model.fit(
                X_text_train, y_train,
                eval_set=[(X_text_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
            )
            
            self.models['text'].append(text_model)
            cv_predictions['text'][val_idx] = text_model.predict_proba(X_text_val)[:, 1]
            
            # Metadata-only model
            print("Training metadata model...")
            X_meta_train, X_meta_val = X_features['meta'][train_idx], X_features['meta'][val_idx]
            
            meta_model = lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                max_depth=8,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED
            )
            
            meta_model.fit(
                X_meta_train, y_train,
                eval_set=[(X_meta_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
            )
            
            self.models['meta'].append(meta_model)
            cv_predictions['meta'][val_idx] = meta_model.predict_proba(X_meta_val)[:, 1]
            
            # Combined model
            print("Training combined model...")
            X_combined_train, X_combined_val = X_features['combined'][train_idx], X_features['combined'][val_idx]
            
            combined_model = lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                max_depth=8,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
                verbose=-1
            )
            
            combined_model.fit(
                X_combined_train, y_train,
                eval_set=[(X_combined_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
            )
            
            self.models['combined'].append(combined_model)
            cv_predictions['combined'][val_idx] = combined_model.predict_proba(X_combined_val)[:, 1]
        
        # Train meta learner on cross-validation predictions
        print("\nTraining meta learner...")
        X_meta_train = np.column_stack([
            cv_predictions['text'],
            cv_predictions['meta'],
            cv_predictions['combined']
        ])
        
        self.meta_learner = LogisticRegression(
            class_weight='balanced',
            random_state=RANDOM_SEED
        )
        self.meta_learner.fit(X_meta_train, y)
        
        # Save models
        for model_type, model_list in self.models.items():
            for i, model in enumerate(model_list):
                model_path = os.path.join(OUTPUT_DIR, "models", f"{model_type}_model_fold_{i}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        meta_learner_path = os.path.join(OUTPUT_DIR, "models", "meta_learner.pkl")
        with open(meta_learner_path, 'wb') as f:
            pickle.dump(self.meta_learner, f)
            
        print("All models saved to", os.path.join(OUTPUT_DIR, "models"))
    
    def load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        self.models = {
            'text': [],
            'meta': [],
            'combined': []
        }
        
        # Load individual models
        for model_type in self.models.keys():
            for i in range(5):  # 5 folds
                model_path = os.path.join(OUTPUT_DIR, "models", f"{model_type}_model_fold_{i}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_type].append(pickle.load(f))
        
        # Load meta learner
        meta_learner_path = os.path.join(OUTPUT_DIR, "models", "meta_learner.pkl")
        if os.path.exists(meta_learner_path):
            with open(meta_learner_path, 'rb') as f:
                self.meta_learner = pickle.load(f)
        
        # Load other artifacts if not already loaded
        if self.tfidf_vectorizer is None:
            tfidf_path = os.path.join(OUTPUT_DIR, "models", "tfidf_vectorizer.pkl")
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                    
        if self.encoder is None:
            encoder_path = os.path.join(OUTPUT_DIR, "models", "categorical_encoder.pkl")
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                    
        if self.scaler is None:
            scaler_path = os.path.join(OUTPUT_DIR, "models", "numeric_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
        if self.pca is None:
            pca_path = os.path.join(OUTPUT_DIR, "models", "pca_embeddings.pkl")
            if os.path.exists(pca_path):
                with open(pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
    
    def predict(self, X_features):
        """Generate predictions using the ensemble"""
        # Ensure models are loaded
        if not self.models.get('text') or not self.meta_learner:
            self.load_models()
        
        # Generate predictions from base models
        base_preds = []
        
        # Predict with text models
        text_preds = np.zeros(X_features['text'].shape[0])
        for model in self.models['text']:
            text_preds += model.predict_proba(X_features['text'])[:, 1]
        text_preds /= len(self.models['text'])
        base_preds.append(text_preds)
        
        # Predict with metadata models
        meta_preds = np.zeros(X_features['meta'].shape[0])
        for model in self.models['meta']:
            meta_preds += model.predict_proba(X_features['meta'])[:, 1]
        meta_preds /= len(self.models['meta'])
        base_preds.append(meta_preds)
        
        # Predict with combined models
        combined_preds = np.zeros(X_features['combined'].shape[0])
        for model in self.models['combined']:
            combined_preds += model.predict_proba(X_features['combined'])[:, 1]
        combined_preds /= len(self.models['combined'])
        base_preds.append(combined_preds)
        
        # Stack predictions for meta-learner
        stacked_preds = np.column_stack(base_preds)
        
        # Get final predictions
        final_probs = self.meta_learner.predict_proba(stacked_preds)[:, 1]
        final_preds = (final_probs >= 0.5).astype(int)
        
        return final_preds, final_probs
    
    def evaluate(self, y_true, y_pred, y_prob=None, set_name="Unknown"):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        print(f"\n----- {set_name} Set Evaluation -----")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Truthful', 'Deceptive'], 
                    yticklabels=['Truthful', 'Deceptive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {set_name} Set')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"confusion_matrix_{set_name.lower()}.png"))
        
        # Plot ROC curve if probabilities available
        if y_prob is not None:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {set_name} Set')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"roc_curve_{set_name.lower()}.png"))
        
        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': macro_f1
        }
        
        return metrics
    
    def plot_feature_importance(self):
        """Plot feature importance for interpretability"""
        if not self.models.get('combined'):
            print("No trained models found. Cannot plot feature importance.")
            return

        # Get feature importance from the first fold of combined model
        combined_model = self.models['combined'][0]

        # Get all feature names used in the metadata model
        numeric_features = self.feature_names['numeric']
        categorical_features = list(self.feature_names['categorical'])
        meta_features = list(numeric_features) + categorical_features

        # Get corresponding feature importances
        meta_importance = combined_model.feature_importances_[:len(meta_features)]

        # De-duplicate by combining importances of repeated names
        unique_feature_map = {}
        for i, name in enumerate(meta_features):
            if name not in unique_feature_map:
                unique_feature_map[name] = meta_importance[i]
            else:
                unique_feature_map[name] += meta_importance[i]

        # Sort by importance and display top 20
        sorted_features = sorted(unique_feature_map.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:20]

        print("\nTop Metadata Features (by importance):")
        for name, importance in top_features:
            print(f"{name:<30} → {importance:.2f}")

        # Plot bar chart
        feat_names, importances = zip(*top_features)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(importances), y=list(feat_names))
        plt.title("Top 20 Most Important Metadata Features")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "feature_importance_cleaned.png"))
        
    def run_pipeline(self):
        """Run the complete pipeline"""
        start_time = datetime.datetime.now()
        print(f"Starting pipeline at {start_time}")
        
        # 1. Load data
        print("\nLoading data...")
        train_data = self.load_data('train.jsonl')
        val_data = self.load_data('validation.jsonl')
        test_data = self.load_data('test.jsonl')
        
        # 2. Extract features
        print("\nExtracting features from training data...")
        train_df, train_labels = self.extract_features(train_data, is_training=True)
        
        print("\nExtracting features from validation data...")
        val_df, val_labels = self.extract_features(val_data, is_training=False)
        
        print("\nExtracting features from test data...")
        test_df, test_labels = self.extract_features(test_data, is_training=False)
        
        # 3. Prepare features for modeling
        print("\nPreparing features...")
        train_features = self.prepare_features(train_df, is_training=True)
        val_features = self.prepare_features(val_df, is_training=False)
        test_features = self.prepare_features(test_df, is_training=False)
        
        # 4. Train models
        print("\nTraining models...")
        self.train_models(train_features, train_labels)
        
        # 5. Generate predictions
        print("\nGenerating validation predictions...")
        val_preds, val_probs = self.predict(val_features)
        
        print("\nGenerating test predictions...")
        test_preds, test_probs = self.predict(test_features)
        
        # 6. Evaluate models
        print("\nEvaluating models...")
        val_metrics = self.evaluate(val_labels, val_preds, val_probs, "Validation")
        test_metrics = self.evaluate(test_labels, test_preds, test_probs, "Test")
        
        # 7. Plot feature importance
        print("\nPlotting feature importance...")
        self.plot_feature_importance()
        
        # 8. Save metrics
        metrics = {
            'validation': val_metrics,
            'test': test_metrics
        }
        
        with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"\nPipeline completed in {duration}")
        print(f"Results saved to {OUTPUT_DIR}")
        
        return metrics

if __name__ == "__main__":
    detector = DiplomacyDeceptionDetector()
    metrics = detector.run_pipeline()
    
    # Print final results
    print("\n===== FINAL RESULTS =====")
    print("\nValidation Metrics:")
    print(f"Accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Precision: {metrics['validation']['precision']:.4f}")
    print(f"Recall: {metrics['validation']['recall']:.4f}")
    print(f"F1 Score: {metrics['validation']['f1']:.4f}")
    print(f"Macro F1: {metrics['validation']['macro_f1']:.4f}")
    
    print("\nTest Metrics:")
    print(f"Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Precision: {metrics['test']['precision']:.4f}")
    print(f"Recall: {metrics['test']['recall']:.4f}")
    print(f"F1 Score: {metrics['test']['f1']:.4f}")
    print(f"Macro F1: {metrics['test']['macro_f1']:.4f}")
    
    # Explain metrics interpretation
    print("\n===== METRICS INTERPRETATION =====")
    print("Accuracy: Percentage of correctly classified messages (both truthful and deceptive)")
    print("Precision: Of all messages predicted as deceptive, what percentage were actually deceptive")
    print("  - Higher means fewer false positives (truthful messages incorrectly labeled as deceptive)")
    print("Recall: Of all actual deceptive messages, what percentage were correctly identified")
    print("  - Higher means fewer false negatives (missed deceptive messages)")
    print("F1 Score: Harmonic mean of precision and recall for deceptive class")
    print("  - Balances precision and recall in a single metric")
    print("Macro F1: Average F1 score across both classes (truthful and deceptive)")
    print("  - Better reflects performance on imbalanced dataset where deceptive messages are rare")