"""
Validate Autoencoder by testing on train/val split
Loads the AE and fold models, makes predictions on val data, 
and compares MAE with the original validation scores
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import ast
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()

# Dataset classes
class FeatureOnlyDataset(Dataset):
    def __init__(self, features):
        self.features = torch.FloatTensor(features)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

class CNNWithLatentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, latent_features, labels=None, use_attention=True, use_self_attention=False, is_train=False):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.latent_features = latent_features
        self.labels = labels
        self.use_attention = use_attention
        self.use_self_attention = use_self_attention
        self.is_train = is_train
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx] if self.attention_mask is not None else [1] * len(input_ids)
        latent_features = self.latent_features[idx]
        
        input_ids = torch.LongTensor(input_ids)
        if self.use_attention or self.use_self_attention:
            mask = torch.FloatTensor(attention_mask)
        else:
            mask = torch.ones_like(input_ids, dtype=torch.float32)
        latent_features = torch.FloatTensor(latent_features)
        
        if self.labels is not None:
            label = torch.FloatTensor([self.labels[idx]]) if isinstance(self.labels[idx], (int, float)) else torch.FloatTensor(self.labels[idx])
            return input_ids, mask, latent_features, label
        return input_ids, mask, latent_features

# Model architectures
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        if mask is not None:
            attn_mask = (mask == 0)
        else:
            attn_mask = None
        residual = x
        out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters=64, kernel_sizes=[3,4,5], dropout=0.3, latent_dim=32, use_attention_mask=True, use_self_attention=False, self_attention_heads=1, self_attention_dropout=0.1, use_attention_pooling=False, use_linear_output=True, activation='silu', fc_architecture='wide'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.use_attention_mask = use_attention_mask
        self.use_self_attention = use_self_attention
        self.use_attention_pooling = use_attention_pooling
        self.use_linear_output = use_linear_output
        
        self.act_fn = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh()}.get(activation.lower(), nn.SiLU())
        
        if use_self_attention:
            self.self_attention = SelfAttention(embedding_dim + latent_dim, num_heads=self_attention_heads, dropout=self_attention_dropout)
        else:
            self.self_attention = None
        
        self.convolutions = nn.ModuleList([
            nn.Conv1d(embedding_dim + latent_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        if use_attention_pooling:
            self.attention_pools = nn.ModuleList([
                AttentionPooling(num_filters) for _ in kernel_sizes
            ])
        else:
            self.attention_pools = None
        
        self.dropout = nn.Dropout(dropout)
        fc_input_dim = num_filters * len(kernel_sizes)
        
        if fc_architecture == 'original':
            fc_dims = [128, 64]
        elif fc_architecture == 'extra_wide':
            fc_dims = [512, 256, 128]
        elif fc_architecture == 'custom':
            fc_dims = [324, 192, 96]
        else:  # 'wide'
            fc_dims = [256, 128, 64]
        
        fc_layers = []
        prev_dim = fc_input_dim
        
        for i, dim in enumerate(fc_dims):
            act_fn = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh()}.get(activation.lower(), nn.SiLU())
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(act_fn)
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        if use_linear_output:
            fc_layers.append(nn.Linear(prev_dim, 1))
        else:
            fc_layers.append(nn.Linear(prev_dim, 1))
            fc_layers.append(nn.Sigmoid())
        
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, input_ids, attention_mask, latent_features):
        input_ids = torch.clamp(input_ids, 0, self.embedding.num_embeddings - 1)
        embedded = self.embedding(input_ids)
        
        if self.use_attention_mask and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            embedded = embedded * mask
        
        embedded = self.dropout(embedded)
        
        if latent_features.dim() == 2:
            latent_expanded = latent_features.unsqueeze(1).expand(-1, embedded.size(1), -1)
        else:
            latent_expanded = latent_features
        
        x = torch.cat([embedded, latent_expanded], dim=2)
        
        if self.use_self_attention and self.self_attention is not None:
            x = self.self_attention(x, attention_mask)
        
        x = x.transpose(1, 2)
        conv_outputs = [self.act_fn(conv(x)) for conv in self.convolutions]
        
        if self.use_attention_pooling and self.attention_pools is not None:
            pooled = [pool(conv_out.transpose(1, 2)) for pool, conv_out in zip(self.attention_pools, conv_outputs)]
        else:
            pooled = [torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outputs]
        
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze()

class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3, activation='gelu'):
        super().__init__()
        act_fn = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh()}.get(activation.lower(), nn.GELU())
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), act_fn
        )
        act_fn2 = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh()}.get(activation.lower(), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), act_fn2, nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    
    def encode(self, x):
        return self.encoder(x)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    cnn_cfg = config['models']['cnn_ae']
    train_cfg = config['training']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("VALIDATING AUTOENCODER ON TRAIN/VAL SPLIT")
    print(f"{'='*80}\n")
    
    # ============ Load Train Data ============
    print(f"{'='*60}")
    print("Loading training data...")
    print(f"{'='*60}\n")
    
    train_df = pd.read_csv("data/processed/train_features.csv")
    
    train_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in train_df['input_ids']]
    train_attention_mask = [ast.literal_eval(mask) if isinstance(mask, str) else mask for mask in train_df['attention_mask']]
    
    max_len = cnn_cfg['max_seq_length']
    train_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in train_input_ids]
    train_attention_mask = [mask[:max_len] + [0]*(max_len-len(mask[:max_len])) for mask in train_attention_mask]
    
    y = train_df['label'].values
    
    # Load features for AE
    corr_df = pd.read_csv("data/processed/feature_correlations.csv", index_col=0)
    top_n = cnn_cfg.get('top_features', 0)
    top_features = corr_df.nlargest(top_n, 'correlation').index.tolist()
    available_features = [f for f in top_features if f in train_df.columns]
    X_features = train_df[available_features].values
    
    print(f"Total training samples: {len(y)}")
    print(f"Features: {len(available_features)}")
    
    # Load test data
    test_df = pd.read_csv("data/processed/test_features.csv")
    test_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in test_df['input_ids']]
    test_attention_mask = [ast.literal_eval(mask) if isinstance(mask, str) else mask for mask in test_df['attention_mask']]
    test_ids = test_df['example_id'].values
    
    test_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in test_input_ids]
    test_attention_mask = [mask[:max_len] + [0]*(max_len-len(mask[:max_len])) for mask in test_attention_mask]
    
    X_test_features = test_df[available_features].values
    print(f"Test samples: {len(test_ids)}")
    
    # ============ Train/Val Split ============
    print(f"\n{'='*60}")
    print("Performing train/val split...")
    print(f"{'='*60}\n")
    
    X_train_ids, X_val_ids, X_train_mask, X_val_mask, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
        train_input_ids, train_attention_mask, X_features, y,
        test_size=train_cfg['validation_split'],
        random_state=train_cfg['seed']
    )
    
    print(f"Train set: {len(y_train)} samples")
    print(f"Val set: {len(y_val)} samples")
    
    # Standardize
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_val_feat = scaler.transform(X_val_feat)
    X_test_features = scaler.transform(X_test_features)
    
    # ============ Load Autoencoder ============
    print(f"\n{'='*60}")
    print("Loading autoencoder...")
    print(f"{'='*60}\n")
    
    ae_path = Path("models/feature_autoencoder.pth")
    if not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder not found at {ae_path}")
    
    ae_hidden = cnn_cfg.get('hidden_dim', 192)
    ae_latent = cnn_cfg.get('ae_latent_dim', max(32, ae_hidden//2))
    ae_dropout = cnn_cfg.get('dropout', 0.3)
    ae_activation = cnn_cfg.get('ae_activation', 'gelu')
    
    ae = FeatureAutoencoder(
        input_dim=X_train_feat.shape[1],
        hidden_dim=ae_hidden,
        latent_dim=ae_latent,
        dropout=ae_dropout,
        activation=ae_activation
    ).to(device)
    
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    print(f"✓ Autoencoder loaded")
    print(f"  Input dim: {X_train_feat.shape[1]}")
    print(f"  Latent dim: {ae_latent}")
    
    # ============ Extract Latent Features ============
    print(f"\n{'='*60}")
    print("Extracting latent features...")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        train_latent = ae.encode(torch.FloatTensor(X_train_feat).to(device)).cpu().numpy()
        val_latent = ae.encode(torch.FloatTensor(X_val_feat).to(device)).cpu().numpy()
        test_latent = ae.encode(torch.FloatTensor(X_test_features).to(device)).cpu().numpy()
    
    print(f"Train latent shape: {train_latent.shape}")
    print(f"Val latent shape: {val_latent.shape}")
    print(f"Test latent shape: {test_latent.shape}")
    
    # ============ Load Models ============
    print(f"\n{'='*60}")
    print("Loading fold models...")
    print(f"{'='*60}\n")
    
    models = {}
    for fold in [1, 2, 3]:
        model_path = Path(f"models/cnn_ae_fold{fold}_best.pth")
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        cnn = CNNModel(
            vocab_size=cnn_cfg['vocab_size'],
            embedding_dim=cnn_cfg['embedding_dim'],
            num_filters=cnn_cfg.get('num_filters', 64),
            kernel_sizes=cnn_cfg.get('kernel_sizes', [3,4,5]),
            dropout=cnn_cfg['dropout'],
            latent_dim=ae_latent,
            use_attention_mask=cnn_cfg.get('use_attention_mask', True),
            use_self_attention=cnn_cfg.get('use_self_attention', False),
            self_attention_heads=cnn_cfg.get('self_attention_heads', 1),
            self_attention_dropout=cnn_cfg.get('self_attention_dropout', 0.1),
            use_attention_pooling=cnn_cfg.get('use_attention_pooling', False),
            use_linear_output=cnn_cfg.get('use_linear_output', True),
            activation=cnn_cfg.get('cnn_activation', 'silu'),
            fc_architecture=cnn_cfg.get('fc_architecture', 'wide')
        ).to(device)
        
        cnn.load_state_dict(torch.load(model_path, map_location=device))
        cnn.eval()
        models[fold] = cnn
        print(f"✓ Loaded fold {fold} model")
    
    # ============ Create Val DataLoader ============
    val_dataset = CNNWithLatentDataset(
        X_val_ids, 
        X_val_mask, 
        val_latent,
        labels=y_val,
        use_attention=cnn_cfg.get('use_attention_mask', True),
        use_self_attention=cnn_cfg.get('use_self_attention', False),
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=cnn_cfg['batch_size'])
    
    # ============ Get Predictions and Calculate MAE ============
    print(f"\n{'='*60}")
    print("Getting predictions on validation data...")
    print(f"{'='*60}\n")
    
    val_preds = {}
    val_maes = {}
    
    with torch.no_grad():
        for fold, model in models.items():
            fold_preds = []
            fold_labels = []
            
            for batch in val_loader:
                input_ids, mask, latent_features, label = batch
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                latent_features = latent_features.to(device)
                label = label.to(device)
                
                preds = model(input_ids, mask, latent_features)
                fold_preds.append(preds.cpu().numpy())
                fold_labels.append(label.cpu().numpy())
            
            fold_preds = np.concatenate(fold_preds)
            fold_labels = np.concatenate(fold_labels).flatten()
            
            # Clamp to [0, 1]
            fold_preds = np.clip(fold_preds, 0, 1)
            
            # Calculate MAE
            mae = np.mean(np.abs(fold_preds - fold_labels))
            
            val_preds[fold] = fold_preds
            val_maes[fold] = mae
            
            print(f"Fold {fold}:")
            print(f"  Predictions shape: {fold_preds.shape}")
            print(f"  Val MAE: {mae:.4f}")
    
    # ============ Get Test Predictions ============
    print(f"\n{'='*60}")
    print("Getting predictions on test data...")
    print(f"{'='*60}\n")
    
    test_dataset = CNNWithLatentDataset(
        test_input_ids,
        test_attention_mask,
        test_latent,
        use_attention=cnn_cfg.get('use_attention_mask', True),
        use_self_attention=cnn_cfg.get('use_self_attention', False),
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=cnn_cfg['batch_size'])
    
    test_preds = {}
    
    with torch.no_grad():
        for fold, model in models.items():
            fold_test_preds = []
            
            for batch in test_loader:
                input_ids, mask, latent_features = batch
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                latent_features = latent_features.to(device)
                
                preds = model(input_ids, mask, latent_features)
                fold_test_preds.append(preds.cpu().numpy())
            
            fold_test_preds = np.concatenate(fold_test_preds)
            fold_test_preds = np.clip(fold_test_preds, 0, 1)
            
            test_preds[fold] = fold_test_preds
            
            print(f"Fold {fold}: Test predictions shape {fold_test_preds.shape}")
    
    # ============ Compute Weighted Ensemble ============
    print(f"\n{'='*60}")
    print("Computing weighted ensemble on validation data...")
    print(f"{'='*60}\n")
    
    original_mae = {1: 0.1564, 2: 0.1568, 3: 0.1548}
    
    # Use inverse-MAE weighting
    inverse_mae_weights = {fold: 1.0 / mae for fold, mae in original_mae.items()}
    weight_sum = sum(inverse_mae_weights.values())
    normalized_weights = {fold: w / weight_sum for fold, w in inverse_mae_weights.items()}
    
    print("Ensemble weights (inverse-MAE):")
    for fold in [1, 2, 3]:
        print(f"  Fold {fold}: {normalized_weights[fold]:.4f}")
    
    # Get validation labels (same for all folds)
    val_labels = y_val
    
    # Compute ensemble predictions
    ensemble_preds = np.zeros_like(val_preds[1])
    for fold in [1, 2, 3]:
        ensemble_preds += normalized_weights[fold] * val_preds[fold]
    
    # Clamp to [0, 1]
    ensemble_preds = np.clip(ensemble_preds, 0, 1)
    
    # Calculate ensemble MAE
    ensemble_mae = np.mean(np.abs(ensemble_preds - val_labels))
    
    print(f"\n✓ Ensemble MAE on validation data: {ensemble_mae:.4f}")
    print(f"  Ensemble predictions - Mean: {ensemble_preds.mean():.4f}, Std: {ensemble_preds.std():.4f}")
    
    # ============ Compare with Original MAE ============
    print(f"\n{'='*60}")
    print("COMPARISON WITH ORIGINAL VALIDATION MAE")
    print(f"{'='*60}\n")
    
    print(f"{'Model':<15} {'Original MAE':<18} {'Current MAE':<18} {'Difference':<18}")
    print("-" * 75)
    
    all_match = True
    for fold in [1, 2, 3]:
        orig = original_mae[fold]
        curr = val_maes[fold]
        diff = abs(orig - curr)
        
        if diff > 0.05:
            all_match = False
        
        print(f"Fold {fold:<10} {orig:<18.4f} {curr:<18.4f} {diff:<18.4f}")
    
    print("-" * 75)
    ensemble_diff = abs(original_mae[1] - ensemble_mae)  # Compare ensemble to fold1 as reference
    print(f"{'Ensemble':<15} {'N/A':<18} {ensemble_mae:<18.4f} {'---':<18}")
    print("-" * 75)
    
    # ============ Summary ============
    print(f"\n{'='*60}")
    print(f"✅ VALIDATION RESULTS")
    print(f"{'='*60}")
    
    if all_match:
        print(f"\n✓ Individual fold MAE values match original values closely!")
    else:
        print(f"\n⚠️  Some fold MAE values differ from original by > 0.05")
    
    print(f"\n✓ Ensemble MAE on validation data: {ensemble_mae:.4f}")
    print(f"  - Fold 1 MAE: {val_maes[1]:.4f}")
    print(f"  - Fold 2 MAE: {val_maes[2]:.4f}")
    print(f"  - Fold 3 MAE: {val_maes[3]:.4f}")
    
    if ensemble_mae < min(val_maes[1], val_maes[2], val_maes[3]):
        print(f"\n✓ Ensemble performs BETTER than any individual fold!")
    else:
        print(f"\n⚠️  Ensemble performance is comparable to individual folds")
    
    # ============ Compute Test Ensemble ============
    print(f"\n{'='*60}")
    print("Computing weighted ensemble on test data...")
    print(f"{'='*60}\n")
    
    test_ensemble_preds = np.zeros_like(test_preds[1])
    for fold in [1, 2, 3]:
        test_ensemble_preds += normalized_weights[fold] * test_preds[fold]
    
    test_ensemble_preds = np.clip(test_ensemble_preds, 0, 1)
    
    print(f"✓ Test ensemble predictions shape: {test_ensemble_preds.shape}")
    print(f"  Mean: {test_ensemble_preds.mean():.4f}, Std: {test_ensemble_preds.std():.4f}")
    
    # ============ Save Test Predictions ============
    print(f"\n{'='*60}")
    print("Saving test predictions...")
    print(f"{'='*60}\n")
    
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble test predictions
    test_ensemble_df = pd.DataFrame({
        'example_id': test_ids,
        'label': test_ensemble_preds
    })
    test_ensemble_path = output_dir / "validate_test_ensemble_preds.csv"
    test_ensemble_df.to_csv(test_ensemble_path, index=False)
    print(f"✓ Test ensemble predictions saved: {test_ensemble_path}")
    
    # Save individual fold test predictions
    for fold in [1, 2, 3]:
        fold_df = pd.DataFrame({
            'example_id': test_ids,
            'label': np.clip(test_preds[fold], 0, 1)
        })
        fold_path = output_dir / f"validate_test_fold{fold}_preds.csv"
        fold_df.to_csv(fold_path, index=False)
        print(f"✓ Test fold {fold} predictions saved: {fold_path}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

