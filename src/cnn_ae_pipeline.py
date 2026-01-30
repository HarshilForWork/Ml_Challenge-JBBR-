"""
AE+CNN Model Training Script
Trains an autoencoder on engineered features, extracts latent features, then trains a CNN using those features.
All params are loaded from config.yaml. Logs as a new MLflow experiment.

OVERFITTING PREVENTION MEASURES:
✓ Train/val split BEFORE autoencoder training (prevents feature leakage)
✓ Scaler fit ONLY on training data (prevents normalization leakage)
✓ Autoencoder has separate train/val split with early stopping (prevents AE overfitting)
✓ AdamW with weight_decay=1e-4 for both AE and CNN (L2 regularization)
✓ Output clamping to [0,1] during TRAINING (not just at end) - helps model learn bounds
✓ Gradient clipping on both AE and CNN (prevents instability)
✓ Validation monitoring for early stopping on CNN training
"""
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('Agg')

load_dotenv()

# Dataset for features only (AE)
class FeatureOnlyDataset(Dataset):
    def __init__(self, features):
        self.features = torch.FloatTensor(features)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

# Data Augmentation Functions
class DataAugmentation:
    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size
        self.token_dropout = config.get('token_dropout', False)
        self.token_dropout_prob = config.get('token_dropout_prob', 0.1)
        self.token_replacement = config.get('token_replacement', False)
        self.token_replacement_prob = config.get('token_replacement_prob', 0.05)
        self.feature_noise = config.get('feature_noise', False)
        self.feature_noise_std = config.get('feature_noise_std', 0.01)
        self.augmentation_prob = config.get('augmentation_prob', 0.5)
    
    def augment_tokens(self, input_ids, attention_mask):
        """Apply token-level augmentation (dropout and replacement)"""
        if random.random() > self.augmentation_prob:
            return input_ids, attention_mask
        
        input_ids = input_ids.copy()
        attention_mask = attention_mask.copy()
        
        # Token dropout: set random non-padding tokens to padding (0)
        if self.token_dropout:
            for i in range(len(input_ids)):
                if attention_mask[i] == 1 and random.random() < self.token_dropout_prob:
                    input_ids[i] = 0
                    attention_mask[i] = 0
        
        # Token replacement: replace random non-padding tokens with random vocab tokens
        if self.token_replacement:
            for i in range(len(input_ids)):
                if attention_mask[i] == 1 and random.random() < self.token_replacement_prob:
                    input_ids[i] = random.randint(1, self.vocab_size - 1)
        
        return input_ids, attention_mask
    
    def augment_features(self, features):
        """Add gaussian noise to features"""
        if not self.feature_noise or random.random() > self.augmentation_prob:
            return features
        
        noise = np.random.normal(0, self.feature_noise_std, features.shape)
        return features + noise

# Dataset for CNN (tokens + attention_mask + AE features + label)
class CNNWithLatentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, latent_features, labels=None, use_attention=True, use_self_attention=False, augmentation=None, is_train=False):
        self.input_ids = input_ids  # Keep as list for augmentation
        self.attention_mask = attention_mask  # Keep as list for augmentation
        self.latent_features = latent_features  # Keep as numpy array for augmentation
        self.labels = labels
        self.use_attention = use_attention
        self.use_self_attention = use_self_attention
        self.augmentation = augmentation
        self.is_train = is_train
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx] if self.attention_mask is not None else [1] * len(input_ids)
        latent_features = self.latent_features[idx]
        
        # Apply augmentation during training
        if self.is_train and self.augmentation is not None:
            input_ids, attention_mask = self.augmentation.augment_tokens(input_ids.copy(), attention_mask.copy())
            latent_features = self.augmentation.augment_features(latent_features.copy())
        
        # Convert to tensors
        input_ids = torch.LongTensor(input_ids)
        if self.use_attention or self.use_self_attention:
            mask = torch.FloatTensor(attention_mask)
        else:
            mask = torch.ones_like(input_ids, dtype=torch.float32)
        latent_features = torch.FloatTensor(latent_features)
        
        if self.labels is not None:
            label = torch.FloatTensor([self.labels[idx]]) if isinstance(self.labels[idx], (int, float)) else torch.FloatTensor(self.labels[idx])
            return input_ids, mask, latent_features, label.squeeze()
        return input_ids, mask, latent_features

class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    def encode(self, x):
        return self.encoder(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        if mask is not None:
            # mask: (batch, seq_len) with 1 for valid, 0 for pad
            attn_mask = (mask == 0)
        else:
            attn_mask = None
        # Self-attention with residual connection
        residual = x
        out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)  # Add & Norm
        return out

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
    def forward(self, x):
        # x: (batch, seq_len, features)
        attn_weights = torch.softmax(self.attention(x), dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, features)
        return pooled

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters=64, kernel_sizes=[3,4,5], dropout=0.3, latent_dim=32, use_attention_mask=True, use_self_attention=False, self_attention_heads=1, self_attention_dropout=0.1, use_attention_pooling=False, use_linear_output=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.use_attention_mask = use_attention_mask
        self.use_self_attention = use_self_attention
        self.use_attention_pooling = use_attention_pooling
        self.use_linear_output = use_linear_output
        if use_self_attention:
            self.self_attention = SelfAttention(embedding_dim + latent_dim, num_heads=self_attention_heads, dropout=self_attention_dropout)
        else:
            self.self_attention = None
        self.convolutions = nn.ModuleList([
            nn.Conv1d(embedding_dim + latent_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        if use_attention_pooling:
            # Attention pooling over sequence dimension for each conv output
            self.attention_pools = nn.ModuleList([
                AttentionPooling(num_filters) for _ in kernel_sizes
            ])
        else:
            self.attention_pools = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        fc_input_dim = num_filters * len(kernel_sizes)
        # Conditional final activation: linear or sigmoid
        if use_linear_output:
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1), nn.Sigmoid()
            )
    def forward(self, input_ids, attention_mask, latent_features):
        input_ids = torch.clamp(input_ids, 0, self.embedding.num_embeddings - 1)
        embedded = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        if self.use_attention_mask and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            embedded = embedded * mask
        embedded = self.dropout(embedded)
        if latent_features.dim() == 2:
            latent_expanded = latent_features.unsqueeze(1).expand(-1, embedded.size(1), -1)
        else:
            latent_expanded = latent_features
        x = torch.cat([embedded, latent_expanded], dim=2)
        # Self-attention block (optional)
        if self.use_self_attention and self.self_attention is not None:
            # ALWAYS pass attention_mask to self-attention to prevent attending to padding
            x = self.self_attention(x, attention_mask)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        conv_outputs = [self.relu(conv(x)) for conv in self.convolutions]
        # Pooling: attention pooling or max pooling
        if self.use_attention_pooling and self.attention_pools is not None:
            # Transpose back for attention pooling: (batch, seq_len, channels)
            pooled = [self.attention_pools[i](c.transpose(1, 2)) for i, c in enumerate(conv_outputs)]
        else:
            # Max pooling over sequence dimension
            pooled = [torch.max(c, dim=2)[0] for c in conv_outputs]
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze()

def train_autoencoder(ae, train_loader, val_loader, epochs, lr, device, early_stopping=True, patience=5):
    """Train AE with optional validation monitoring and early stopping to prevent overfitting."""
    ae = ae.to(device)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        ae.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        ae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = ae(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"AE Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Early stopping (if enabled)
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⚠️  AE Early stopping at epoch {epoch+1}")
                    break
    return ae

def apply_mixup(inputs_ids, attention_masks, latent_features, labels, alpha=0.2):
    """Apply mixup data augmentation"""
    batch_size = len(labels)
    if batch_size < 2:
        return inputs_ids, attention_masks, latent_features, labels
    
    indices = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)
    
    # Mix latent features and labels
    mixed_latent = lam * latent_features + (1 - lam) * latent_features[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    # For tokens, we can't really interpolate, so we randomly choose based on lambda
    # If lam > 0.5, use original, else use shuffled
    if lam > 0.5:
        return inputs_ids, attention_masks, mixed_latent, mixed_labels
    else:
        return inputs_ids[indices], attention_masks[indices], mixed_latent, mixed_labels

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_name = "cnn_ae"
    cnn_cfg = config['models']['cnn_ae']
    train_cfg = config['training']
    
    # Data augmentation config
    use_data_augmentation = cnn_cfg.get('use_data_augmentation', False)
    augmentation_config = cnn_cfg.get('augmentation', {})
    use_mixup = augmentation_config.get('mixup', False) if use_data_augmentation else False
    mixup_alpha = augmentation_config.get('mixup_alpha', 0.2)
    
    # Overfitting prevention toggles
    prevent_overfitting = cnn_cfg.get('prevent_overfitting', True)
    ae_early_stopping = cnn_cfg.get('ae_early_stopping', True) if prevent_overfitting else False
    ae_early_stopping_patience = cnn_cfg.get('ae_early_stopping_patience', 5)
    ae_train_val_split = cnn_cfg.get('ae_train_val_split', True) if prevent_overfitting else False
    ae_train_val_ratio = cnn_cfg.get('ae_train_val_ratio', 0.9)
    enforce_weight_decay = cnn_cfg.get('enforce_weight_decay', True) if prevent_overfitting else False
    min_weight_decay = cnn_cfg.get('min_weight_decay', 0.0001)
    clamp_outputs_during_training = cnn_cfg.get('clamp_outputs_during_training', True) if prevent_overfitting else False
    scaler_fit_train_only = cnn_cfg.get('scaler_fit_train_only', True) if prevent_overfitting else False
    
    print(f"{'='*60}")
    print(f"Data Augmentation Status (use_data_augmentation={use_data_augmentation})")
    if use_data_augmentation:
        print(f"  Token Dropout: {augmentation_config.get('token_dropout', False)} (p={augmentation_config.get('token_dropout_prob', 0.1)})")
        print(f"  Token Replacement: {augmentation_config.get('token_replacement', False)} (p={augmentation_config.get('token_replacement_prob', 0.05)})")
        print(f"  Feature Noise: {augmentation_config.get('feature_noise', False)} (std={augmentation_config.get('feature_noise_std', 0.01)})")
        print(f"  Mixup: {use_mixup} (alpha={mixup_alpha})")
        print(f"  Augmentation Probability: {augmentation_config.get('augmentation_prob', 0.5)}")
    print(f"{'='*60}")
    print(f"Overfitting Prevention Status (prevent_overfitting={prevent_overfitting})")
    print(f"  AE Early Stopping: {ae_early_stopping} (patience={ae_early_stopping_patience})")
    print(f"  AE Train/Val Split: {ae_train_val_split} (ratio={ae_train_val_ratio})")
    print(f"  Enforce Weight Decay: {enforce_weight_decay} (min={min_weight_decay})")
    print(f"  Clamp Outputs During Training: {clamp_outputs_during_training}")
    print(f"  Scaler Fit Train Only: {scaler_fit_train_only}")
    print(f"{'='*60}")
    ae_input_type = cnn_cfg.get('ae_input_type', 'features')  # 'features' or 'features+embeddings'
    use_attention_mask = cnn_cfg.get('use_attention_mask', True)  # Toggle for attention masking
    use_self_attention = cnn_cfg.get('use_self_attention', False)
    self_attention_heads = cnn_cfg.get('self_attention_heads', 1)
    self_attention_dropout = cnn_cfg.get('self_attention_dropout', 0.1)
    use_attention_pooling = cnn_cfg.get('use_attention_pooling', False)
    use_adamw = cnn_cfg.get('use_adamw', True)
    weight_decay = cnn_cfg.get('weight_decay', 0.0)
    use_linear_output = cnn_cfg.get('use_linear_output', True)
    # Data loading
    train_df = pd.read_csv("data/processed/train_features.csv")
    test_df = pd.read_csv("data/processed/test_features.csv")
    import ast
    train_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in train_df['input_ids']]
    test_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in test_df['input_ids']]
    
    # Load attention_mask
    train_attention_mask = [ast.literal_eval(mask) if isinstance(mask, str) else mask for mask in train_df['attention_mask']]
    test_attention_mask = [ast.literal_eval(mask) if isinstance(mask, str) else mask for mask in test_df['attention_mask']]
    
    max_len = cnn_cfg['max_seq_length']
    train_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in train_input_ids]
    test_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in test_input_ids]
    
    # Pad attention_mask to same length
    train_attention_mask = [mask[:max_len] + [0]*(max_len-len(mask[:max_len])) for mask in train_attention_mask]
    test_attention_mask = [mask[:max_len] + [0]*(max_len-len(mask[:max_len])) for mask in test_attention_mask]
    
    y = train_df['label'].values
    test_ids = test_df['example_id'].values
    top_n = cnn_cfg.get('top_features', 0)
    if top_n <= 0:
        raise ValueError('top_features must be > 0 for AE pipeline')
    corr_df = pd.read_csv("data/processed/feature_correlations.csv", index_col=0)
    top_features = corr_df.nlargest(top_n, 'correlation').index.tolist()
    available_features = [f for f in top_features if f in train_df.columns]
    X_features = train_df[available_features].values
    X_test_features = test_df[available_features].values
    
    print(f"Using {len(available_features)} top engineered features for autoencoder")
    if use_attention_mask:
        print(f"✓ Using attention_mask for MASKED POOLING in CNN (zeros out padding embeddings)")
    else:
        print(f"✗ Attention masking DISABLED - using raw embeddings without masking")
    
    # Note: Standardization will happen AFTER train/val split to prevent leakage
    # (scaler will be fit ONLY on training data)
    # Always compute mean token embeddings for all samples
    temp_vocab_size = max(max(seq) for seq in train_input_ids) + 1
    temp_emb_dim = cnn_cfg['embedding_dim']
    temp_emb = nn.Embedding(temp_vocab_size, temp_emb_dim, padding_idx=0)
    with torch.no_grad():
        train_embs = []
        for ids in train_input_ids:
            arr = torch.LongTensor(ids)
            emb = temp_emb(arr)
            mask = (arr != 0).float().unsqueeze(1)
            mean_emb = (emb * mask).sum(0) / mask.sum() if mask.sum() > 0 else emb.mean(0)
            train_embs.append(mean_emb.numpy())
        test_embs = []
        for ids in test_input_ids:
            arr = torch.LongTensor(ids)
            emb = temp_emb(arr)
            mask = (arr != 0).float().unsqueeze(1)
            mean_emb = (emb * mask).sum(0) / mask.sum() if mask.sum() > 0 else emb.mean(0)
            test_embs.append(mean_emb.numpy())
    if ae_input_type == 'features+embeddings':
        X_features = np.concatenate([X_features, np.stack(train_embs)], axis=1)
        X_test_features = np.concatenate([X_test_features, np.stack(test_embs)], axis=1)
    # Train/val split BEFORE autoencoder training to prevent data leakage
    X_train_ids, X_val_ids, X_train_mask, X_val_mask, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
        train_input_ids, train_attention_mask, X_features, y,
        test_size=train_cfg['validation_split'],
        random_state=train_cfg['seed']
    )
    print(f"⚠️  IMPORTANT: Train/val split performed BEFORE autoencoder training to prevent data leakage")
    print(f"   Train set: {X_train_feat.shape[0]} samples | Val set: {X_val_feat.shape[0]} samples")
    
    # Standardize: fit scaler based on toggle
    scaler = StandardScaler()
    if scaler_fit_train_only:
        X_train_feat = scaler.fit_transform(X_train_feat)
        X_val_feat = scaler.transform(X_val_feat)
        X_test_features = scaler.transform(X_test_features)
    else:
        # Legacy: fit on all data (not recommended, but available for testing)
        all_features = np.vstack([X_train_feat, X_val_feat])
        scaler.fit(all_features)
        X_train_feat = scaler.transform(X_train_feat)
        X_val_feat = scaler.transform(X_val_feat)
        X_test_features = scaler.transform(X_test_features)
    
    # Create AE train/val split (if enabled)
    if ae_train_val_split:
        ae_train_size = int(len(X_train_feat) * ae_train_val_ratio)
        ae_train_feat = X_train_feat[:ae_train_size]
        ae_val_feat = X_train_feat[ae_train_size:]
        print(f"⚠️  AE train/val split: {ae_train_feat.shape[0]} / {ae_val_feat.shape[0]} (prevent AE overfitting)")
    else:
        # Use full training data for AE
        ae_train_feat = X_train_feat
        ae_val_feat = X_val_feat
        print(f"⚠️  AE using full training data (no AE train/val split)")
    
    # AE config
    ae_hidden = cnn_cfg.get('hidden_dim', 192)
    # AE latent dim logic
    ae_compression_mode = cnn_cfg.get('ae_compression_mode', 'auto')
    input_dim = X_train_feat.shape[1]
    if ae_compression_mode == 'compress':
        ae_latent = min(cnn_cfg.get('ae_latent_dim', input_dim), input_dim)
    elif ae_compression_mode == 'expand':
        ae_latent = max(cnn_cfg.get('ae_latent_dim', input_dim), input_dim)
    else:  # auto
        ae_latent = cnn_cfg.get('ae_latent_dim', max(32, ae_hidden//2))
    ae_dropout = cnn_cfg.get('dropout', 0.3)
    ae_epochs = cnn_cfg.get('ae_epochs', 30)
    ae_lr = cnn_cfg.get('ae_lr', 1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Train AE
    print(f"\nTraining feature autoencoder on: {ae_input_type}")
    print(f"AE latent dimension: {ae_latent}")
    print(f"Embedding dim: {cnn_cfg['embedding_dim']}, Total: {cnn_cfg['embedding_dim'] + ae_latent}")
    print(f"Self-attention heads: {cnn_cfg.get('self_attention_heads', 1)}")
    print(f"Divisibility check: {(cnn_cfg['embedding_dim'] + ae_latent) % cnn_cfg.get('self_attention_heads', 1) == 0}")
    ae = FeatureAutoencoder(input_dim=X_train_feat.shape[1], hidden_dim=ae_hidden, latent_dim=ae_latent, dropout=ae_dropout)
    ae_train_loader = DataLoader(FeatureOnlyDataset(ae_train_feat), batch_size=64, shuffle=True)
    ae_val_loader = DataLoader(FeatureOnlyDataset(ae_val_feat), batch_size=64, shuffle=False)
    ae = train_autoencoder(ae, ae_train_loader, ae_val_loader, ae_epochs, ae_lr, device, 
                          early_stopping=ae_early_stopping, patience=ae_early_stopping_patience)
    # Extract latent features
    ae.eval()
    with torch.no_grad():
        train_latent = ae.encode(torch.FloatTensor(X_train_feat).to(device)).cpu().numpy()
        val_latent = ae.encode(torch.FloatTensor(X_val_feat).to(device)).cpu().numpy()
        test_latent = ae.encode(torch.FloatTensor(X_test_features).to(device)).cpu().numpy()
    # Check for NaN in latent features
    if np.isnan(train_latent).any():
        print("⚠️  WARNING: NaN detected in train_latent features!")
    if np.isnan(val_latent).any():
        print("⚠️  WARNING: NaN detected in val_latent features!")
    # Print AE input/output/latent dimensions and a sample of latent features
    print(f"\nAE input dim: {ae.encoder[0].in_features}")
    print(f"AE latent dim: {ae.encoder[-2].out_features}")
    print(f"AE output dim: {ae.decoder[-1].out_features}")
    print(f"Latent feature array shape: {train_latent.shape}")
    print("Sample AE latent features (first 3 samples, first 8 dims):")
    print(np.round(train_latent[:3, :8], 4))
    # Note: latent features stay 2D (batch, latent_dim) - model expands them dynamically
    # CNN config
    max_token_id = int(max(max(seq) for seq in X_train_ids))
    vocab_size = max_token_id + 1
    
    # Initialize data augmentation
    augmentation = None
    if use_data_augmentation:
        augmentation = DataAugmentation(augmentation_config, vocab_size)
        print(f"\n✓ Data augmentation initialized")
    if enforce_weight_decay and weight_decay <= 0:
        weight_decay = min_weight_decay
        print(f"⚠️  weight_decay was {cnn_cfg.get('weight_decay', 0.0)} - enforcing minimum {weight_decay}")
    
    cnn = CNNModel(
        vocab_size=vocab_size,
        embedding_dim=cnn_cfg['embedding_dim'],
        num_filters=cnn_cfg.get('num_filters', 64),
        kernel_sizes=cnn_cfg.get('kernel_sizes', [3,4,5]),
        dropout=cnn_cfg['dropout'],
        latent_dim=ae_latent,
        use_attention_mask=use_attention_mask,
        use_self_attention=use_self_attention,
        self_attention_heads=self_attention_heads,
        self_attention_dropout=self_attention_dropout,
        use_attention_pooling=use_attention_pooling,
        use_linear_output=use_linear_output
    ).to(device)
    # Datasets
    train_dataset = CNNWithLatentDataset(X_train_ids, X_train_mask, train_latent, y_train, use_attention=use_attention_mask, use_self_attention=use_self_attention, augmentation=augmentation, is_train=True)
    val_dataset = CNNWithLatentDataset(X_val_ids, X_val_mask, val_latent, y_val, use_attention=use_attention_mask, use_self_attention=use_self_attention, augmentation=None, is_train=False)
    test_dataset = CNNWithLatentDataset(test_input_ids, test_attention_mask, test_latent, use_attention=use_attention_mask, use_self_attention=use_self_attention, augmentation=None, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=cnn_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cnn_cfg['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=cnn_cfg['batch_size'])
    # MLflow setup
    experiment_name = f"prompt-quality-{model_name}"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_name}-run"):
        params = {
            'model': model_name,
            'vocab_size': vocab_size,
            'embedding_dim': cnn_cfg['embedding_dim'],
            'max_seq_length': cnn_cfg['max_seq_length'],
            'num_filters': cnn_cfg.get('num_filters', 64),
            'kernel_sizes': str(cnn_cfg.get('kernel_sizes', [3,4,5])),
            'dropout': cnn_cfg['dropout'],
            'batch_size': cnn_cfg['batch_size'],
            'learning_rate': cnn_cfg['learning_rate'],
            'num_epochs': cnn_cfg['num_epochs'],
            'ae_hidden_dim': ae_hidden,
            'ae_latent_dim': ae_latent,
            'ae_epochs': ae_epochs,
            'ae_lr': ae_lr,
            'top_features': top_n,
            'ae_input_type': ae_input_type,
            'loss_function': train_cfg.get('loss_function', 'MSELoss'),
            'use_scheduler': train_cfg.get('use_scheduler', False),
            'use_attention_mask': use_attention_mask,
            'attention_mask_strategy': 'masked_pooling' if use_attention_mask else 'disabled',
            'use_self_attention': use_self_attention,
            'self_attention_heads': self_attention_heads,
            'self_attention_dropout': self_attention_dropout,
            'use_attention_pooling': use_attention_pooling,
            'use_adamw': use_adamw,
            'weight_decay': weight_decay if use_adamw else 0.0,
            'optimizer': 'AdamW' if use_adamw else 'Adam',
            'use_linear_output': use_linear_output,
            'output_activation': 'linear+clip' if use_linear_output else 'sigmoid',
            'use_data_augmentation': use_data_augmentation,
            'token_dropout': augmentation_config.get('token_dropout', False) if use_data_augmentation else False,
            'token_dropout_prob': augmentation_config.get('token_dropout_prob', 0.0),
            'token_replacement': augmentation_config.get('token_replacement', False) if use_data_augmentation else False,
            'token_replacement_prob': augmentation_config.get('token_replacement_prob', 0.0),
            'feature_noise': augmentation_config.get('feature_noise', False) if use_data_augmentation else False,
            'feature_noise_std': augmentation_config.get('feature_noise_std', 0.0),
            'mixup': use_mixup,
            'mixup_alpha': mixup_alpha if use_mixup else 0.0,
            'augmentation_prob': augmentation_config.get('augmentation_prob', 0.0)
        }
        mlflow.log_params(params)
        # Loss, optimizer, scheduler
        loss_fn = train_cfg.get('loss_function', 'MSELoss')
        criterion = nn.L1Loss() if loss_fn == 'L1Loss' else nn.MSELoss()
        print(f"Loss function: {loss_fn}")
        if use_adamw:
            print(f"Optimizer: AdamW (weight_decay={weight_decay})")
            optimizer = torch.optim.AdamW(cnn.parameters(), lr=cnn_cfg['learning_rate'], weight_decay=weight_decay)
        else:
            print(f"Optimizer: Adam")
            optimizer = torch.optim.Adam(cnn.parameters(), lr=cnn_cfg['learning_rate'])
        scheduler = None
        if train_cfg.get('use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=train_cfg.get('scheduler_factor', 0.5),
                patience=train_cfg.get('scheduler_patience', 5),
                verbose=True
            )
        best_val_mae = float('inf')
        patience_counter = 0
        for epoch in range(cnn_cfg['num_epochs']):
            cnn.train()
            total_loss = 0
            preds, targets = [], []
            for batch_ids, batch_mask, batch_latent, batch_labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_latent = batch_latent.to(device)
                batch_labels = batch_labels.to(device)
                
                # Apply mixup augmentation if enabled
                if use_mixup and use_data_augmentation:
                    batch_ids, batch_mask, batch_latent, batch_labels = apply_mixup(
                        batch_ids, batch_mask, batch_latent, batch_labels, alpha=mixup_alpha
                    )
                
                optimizer.zero_grad()
                outputs = cnn(batch_ids, batch_mask, batch_latent)
                # Clamp outputs to [0,1] during training if enabled and using linear output
                if clamp_outputs_during_training and use_linear_output:
                    outputs = torch.clamp(outputs, 0, 1)
                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    print(f"⚠️  NaN in outputs! Batch stats:")
                    print(f"  - batch_latent has NaN: {torch.isnan(batch_latent).any()}")
                    print(f"  - batch_ids range: [{batch_ids.min()}, {batch_ids.max()}]")
                    raise ValueError("NaN detected in model outputs")
                loss = criterion(outputs, batch_labels)
                if torch.isnan(loss):
                    print(f"⚠️  NaN in loss! outputs range: [{outputs.min()}, {outputs.max()}]")
                    raise ValueError("NaN detected in loss")
                loss.backward()
                # Gradient clipping to prevent NaN
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                preds.extend(outputs.detach().cpu().numpy())
                targets.extend(batch_labels.cpu().numpy())
            train_mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
            cnn.eval()
            val_loss, val_mae = 0, 0
            vpreds, vtargets = [], []
            with torch.no_grad():
                for batch_ids, batch_mask, batch_latent, batch_labels in tqdm(val_loader, desc=f"Val {epoch+1}"):
                    batch_ids = batch_ids.to(device)
                    batch_mask = batch_mask.to(device)
                    batch_latent = batch_latent.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = cnn(batch_ids, batch_mask, batch_latent)
                    # Clamp during validation if enabled and using linear output
                    if clamp_outputs_during_training and use_linear_output:
                        outputs = torch.clamp(outputs, 0, 1)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    vpreds.extend(outputs.cpu().numpy())
                    vtargets.extend(batch_labels.cpu().numpy())
            val_loss /= len(val_loader)
            val_mae = np.mean(np.abs(np.array(vpreds) - np.array(vtargets)))
            print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f}, MAE {train_mae:.4f} | Val Loss {val_loss:.4f}, MAE {val_mae:.4f}")
            mlflow.log_metrics({'train_loss': total_loss/len(train_loader), 'train_mae': train_mae, 'val_loss': val_loss, 'val_mae': val_mae}, step=epoch)
            if scheduler is not None:
                scheduler.step(val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                model_path = f'models/{model_name}_best.pth'
                os.makedirs('models', exist_ok=True)
                torch.save(cnn.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                print(f"✓ Best model saved! MAE: {best_val_mae:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= train_cfg['early_stopping_patience']:
                    print("\nEarly stopping triggered!")
                    break
        # Predict
        cnn.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
        cnn.eval()
        preds = []
        with torch.no_grad():
            for batch_ids, batch_mask, batch_latent in tqdm(test_loader, desc="Predict"):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_latent = batch_latent.to(device)
                outputs = cnn(batch_ids, batch_mask, batch_latent)
                # Clamp only if enabled and using linear output (sigmoid already in [0,1])
                if clamp_outputs_during_training and use_linear_output:
                    outputs = torch.clamp(outputs, 0, 1)
                preds.extend(outputs.cpu().numpy())
        preds = np.array(preds)
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame({'example_id': test_ids, 'label': preds})
        output_path = output_dir / f"test-{model_name}.csv"
        output_df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved: {output_path}")
        mlflow.log_artifact(str(output_path))
        mlflow.log_artifact('src/cnn_ae_pipeline.py')  # Log source code as artifact
        mlflow.log_metric('best_val_mae', best_val_mae)
        print(f"\n{'='*60}")
        print(f"✅ AE+CNN Training Complete!")
        print(f"Best Val MAE: {best_val_mae:.4f}")
        print(f"Model: models/{model_name}_best.pth")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
