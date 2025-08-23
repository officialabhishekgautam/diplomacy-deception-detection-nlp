# Diplomacy Deception Detection: Multi-Model NLP Pipeline
A comprehensive machine learning project implementing multiple approaches to detect deceptive communications in the strategic board game Diplomacy. This project compares transformer-based models, sequence labeling approaches, and gradient boosting methods for conversational deception detection.

# 🎯 Project Overview
Deception detection in natural language processing presents unique challenges, particularly in strategic gaming contexts where players may deliberately mislead opponents. This project tackles the QANTA Diplomacy dataset to build robust models that can identify deceptive vs. truthful messages exchanged between players during gameplay.

# 🏆 Key Results
ModelTest AccuracyTest Macro F1Test ROC-AUCDomain-Adaptive RoBERTa91.24%47.71%79.76%BiLSTM + CRF76.14%59.50%-LightGBM + SentenceBERT62.09%50.47%-

# 🚀 Features

Multi-Model Architecture: Three distinct approaches for comprehensive comparison
Rich Feature Engineering: Incorporates game metadata, temporal features, and linguistic patterns
Domain Adaptation: Custom pre-training on Diplomacy-specific text corpus
Advanced Evaluation: Comprehensive metrics including accuracy, precision, recall, F1, and ROC-AUC
Error Analysis: Detailed performance breakdown and failure case analysis

# Models Implemented

## 1. Domain-Adaptive Pre-trained Feature-Rich Transformer

Architecture: RoBERTa with domain-specific pre-training
Features: Contextual embeddings + game metadata + temporal features
Preprocessing: Custom tokenization with special game-specific tokens
Performance: 91.24% accuracy, 79.76% ROC-AUC

## 2. BiLSTM + CRF with Rich Metadata

Architecture: Bidirectional LSTM with Conditional Random Fields
Features: BERT embeddings + extensive metadata integration
Approach: Sequence labeling with structured prediction
Performance: 76.14% accuracy, 59.50% macro F1

## 3. LightGBM + SentenceBERT

Architecture: Gradient boosting with dense sentence embeddings
Features: SentenceBERT representations + engineered features
Approach: Traditional ML with modern embeddings
Performance: 62.09% accuracy, 50.47% macro F1
