# Obfuscated Prompt Quality Prediction - ML & MLOps Challenge

## 📌 Project Overview
This project targets the **Unstop ML Challenge (Udhgam 2.0)**, a supervised regression problem to predict a continuous "prompt quality" score (0-1) from obfuscated token sequences. The core challenge lies in the inability to deanonymize tokens or use pretrained language models (like BERT/GPT) tied to known vocabularies.

This repository implements a **robust MLOps-driven pipeline** where we benchmarked Transformers, RNNs, and CNNs. The **Champion Model (CNN Autoencoder)** achieved a **Mean Absolute Error (MAE) of 0.1560**, significantly outperforming traditional sequence models through a novel fusion of Latent Feature Extraction and Multi-Scale Convolutions.

---

## 🚀 Key Features & Highlights

### 1. Advanced Feature Engineering
Since the tokens are anonymized, we cannot rely on semantics. Instead, we extract high-dimensional signals from the token structure:
- **Statistical Features**: Mean, Std, Skewness, Kurtosis of token IDs.
- **Information Theoretic Features**: Entropy, Gini Coefficient, Repetition Rate to measure prompt complexity.
- **Structural Features**: Sequence length, padding ratios, N-gram diversity (Bigrams/Trigrams).
- **Positional Features**: First/Last token analysis and token distance metrics.

### 2. Champion Model: CNN Autoencoder (MAE: 0.1560)
Our matching-winning architecture uses a novel **Two-Stage Pipeline**:
1. **Latent Feature Extraction (Autoencoder)**: A dense Autoencoder compresses the 40+ statistical features into a robust, denoised scalar representation (Latent Vector). This acts as a trainable feature selector.
2. **Multi-Scale 1D-CNN**: The core regressor. It ingests a concatenation of:
  - **Token Embeddings**: Learned representations of the obfuscated IDs.
  - **Latent Features**: The Autoencoder's compressed output expanded to match sequence length.
  - **Multi-Kernel Convolutions**: Kernels of sizes [3, 4, 5] capture local patterns at different scales (like N-grams).



### 3. Experimental Baselines
We extensively benchmarked **RNNs (LSTM, GRU)** and **Transformers**. While effective, they struggled to generalize on the obfuscated statistical patterns compared to the CNN-AE, which efficiently combined local token patterns with global statistical signals.

### 4. MLOps & Production Readiness
- **Experiment Tracking**: Full integration with **MLflow** to track metrics (MAE, RMSE, Loss), hyperparameters, and model artifacts.
- **Data Version Control (DVC)**: Configuration for **DVC with S3 backend** to version control large datasets (`Raw_data`, `input_data`) and trained models.
- **Config-Driven Development**: Centralized `config.yaml` to manage hyperparameters for all models (CNN, LSTM, Transformer, Hybrid).
- **Reproducibility**: Seed sealing, deterministic data splitting, and environment management.

---

## 🛠️ Tech Stack

### Data Science & Machine Learning
- **Core**: `Python 3.10+`, `NumPy`, `Pandas`
- **Deep Learning**: `PyTorch` (Custom Transformers, Embeddings, Optimizers)
- **Gradient Boosting**: `CatBoost`
- **Scikit-Learn**: preprocessing (`StandardScaler`), model selection (`train_test_split`), metrics (`MAE`, `MSE`)

### MLOps & Engineering
- **Experiment Tracking**: `MLflow`
- **Data Versioning**: `DVC` (Data Version Control)
- **Cloud Storage**: `AWS S3` (via `boto3` for DVC remote)
- **Configuration**: `PyYAML`, `python-dotenv`
- **Data Processing**: `jsonlines`, `tqdm`

---

## 📂 Project Structure

```bash
├── .dvc/                   # DVC configuration
├── .github/                # CI/CD workflows
├── data/
│   ├── processed/          # Feature-engineered datasets (parquet/csv)
│   └── output/             # Submission files and predictions
├── mlruns/                 # MLflow local tracking store
├── models/                 # Saved model checkpoints (.pth, .cbm)
├── src/
│   ├── feature_engineering.py  # Statistical & structural feature extraction
│   ├── transformer_regressor.py # PyTorch Transformer & CatBoost training loop
│   ├── cnn_ae_pipeline.py      # Alternative CNN Autoencoder pipeline
│   ├── transformer.py          # Transformer model definition
│   └── ...                     # Other architectures (LSTM, GRU, CNN)
├── config.yaml             # Centralized hyperparameter configuration
├── requirements.txt        # Project dependencies
├── setup_dvc_mlflow.py     # Setup script for MLOps tools
└── ps.txt                  # Problem Statement
```

---

## ⚙️ Model Architectures

The project supports modular model switching via `config.yaml`. Implemented architectures include:

| Model | Description | Use Case |
| :--- | :--- | :--- |
| **CNN Autoencoder (CNN-AE)** | **Feature AE + 1D-CNN** | **Champion Model (MAE: 0.1560)** - State-of-the-art performance. |
| **Transformer Regressor** | Custom Encoder-only Transformer | Strong baseline, good global context capture. |
| **CatBoost Hybrid** | Gradient Boosting on Embeddings | Strong tabular performance, used for ensembling. |
| **LSTM / GRU / RNN** | RNN-based sequential models | Experimental baselines (Underperformed). |
| **CNN 1D** | Multi-kernel 1D Convolutional Network | Fast baseline, captures local patterns well. |

---

## 📊 Performance Optimization (Training Details)
- **K-Fold Cross Validation**: 5-Fold Stratified Cross-Validation to ensure robust error estimation.
- **Latent Feature Denoising**: Autoencoder pre-training significantly reduced feature variable variance.
- **Loss Function**: `L1Loss` (MAE) directly optimized for the competition metric.
- **Optimization**: `AdamW` optimizer with `ReduceLROnPlateau` scheduler.
- **Regularization**: `Dropout` (0.3), `Weight Decay` (1e-4), and `Gradient Clipping` to prevent exploding gradients.
- **Augmentation**:
  - **MixUp**: Interpolating between samples in the latent space.
  - **Feature Noise**: Adding Gaussian noise to prevent overfitting.

---

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Create a `.env` file for MLflow/AWS credentials (optional for local run).

3. **Run Feature Engineering**
   ```bash
   python src/feature_engineering.py
   ```
   *Extracts features from `train.jsonl`/`test.jsonl` and saves to `data/processed/`.*

4. **Train Champion Model (CNN AE)**
   ```bash
   # Runs the Autoencoder pre-training followed by CNN regression
   python src/cnn_ae_pipeline.py
   ```

5. **View Experiments**
   ```bash
   mlflow ui
   ```
