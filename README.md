# Multimodal Deception Classifier (Video + Text)

This code implements a **multimodal classification pipeline** that fuses **video features** and **text** to predict deception labels. It uses a Transformer-based video encoder and a RoBERTa text encoder, trained with cross-entropy loss.

---

## Components

### Dataset
`MultimodalDeceptionDataset`
- Video features: `(N, T, FEAT_DIM)`
- Video masks: `(N, T)` bool (`True` = valid frame)
- Texts: list of strings
- Labels: `(N,)` int
- Tokenization: RoBERTa with padding/truncation to `TEXT_MAX_LEN`

Each batch returns:
- `video_feats`, `video_mask`
- `input_ids`, `attention_mask`
- `labels`

---

### Video Encoder
`VideoTransformer`
- Linear projection + learnable positional embeddings
- Transformer encoder over time dimension
- Masked mean pooling over valid frames
- Output shape: `(B, EMBED_DIM)`

---

### Multimodal Classifier
`MultiModalClassifier`
- Video encoder + RoBERTa text encoder
- Uses `[CLS]` token for text representation
- Projects text and video to same embedding space
- Concatenates and classifies via MLP
- Supports freezing text encoder

Output:
- `logits`: `(B, num_classes)`
- `loss`: cross-entropy (if labels provided)

---

## Training & Evaluation

- `train_one_epoch`: standard supervised training loop
- `evaluate_model`: computes loss, accuracy, precision, recall, F1, and ROC-AUC
- Positive class is **label 1**

---

## Cross Validation

- 5-fold **StratifiedKFold** on training data only
- Tracks train/validation loss per epoch
- Saves best model per fold based on validation accuracy
- Plots loss curves for each fold

---

## Final Training & Test

- Retrains model on **full training set**
- Plots training loss curve
- Evaluates once on a **held-out test set**

---

## Requirements

- Python 3.9+
- PyTorch
- HuggingFace `transformers`
- NumPy, scikit-learn, matplotlib

