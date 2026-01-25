"""
AE+CNN Model Training Script
Trains an autoencoder on engineered features, extracts latent features, then trains a CNN using those features.
All params are loaded from config.yaml. Logs as a new MLflow experiment.
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

# Dataset for CNN (tokens + AE features + label)
class CNNWithLatentDataset(Dataset):
    def __init__(self, input_ids, latent_features, labels=None):
        self.input_ids = torch.LongTensor(input_ids)
        self.latent_features = torch.FloatTensor(latent_features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        # Return both input_ids and latent_features for each sample
        if self.labels is not None:
            return self.input_ids[idx], self.latent_features[idx], self.labels[idx]
        return self.input_ids[idx], self.latent_features[idx]

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

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters=64, kernel_sizes=[3,4,5], dropout=0.3, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.convolutions = nn.ModuleList([
            nn.Conv1d(embedding_dim + latent_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        fc_input_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, input_ids, latent_features):
        input_ids = torch.clamp(input_ids, 0, self.embedding.num_embeddings - 1)
        embedded = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        embedded = self.dropout(embedded)
        # Expand latent_features to (batch, seq_len, latent_dim)
        if latent_features.dim() == 2:
            latent_expanded = latent_features.unsqueeze(1).expand(-1, embedded.size(1), -1)
        else:
            latent_expanded = latent_features
        # Concatenate along feature dim
        x = torch.cat([embedded, latent_expanded], dim=2)  # (batch, seq_len, emb_dim+latent_dim)
        x = x.transpose(1, 2)  # (batch, emb_dim+latent_dim, seq_len)
        conv_outputs = [self.relu(conv(x)) for conv in self.convolutions]
        pooled = [torch.max(c, dim=2)[0] for c in conv_outputs]
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze()

def train_autoencoder(ae, dataloader, epochs, lr, device):
    ae = ae.to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    ae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"AE Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
    return ae

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model_name = "cnn_ae"
    cnn_cfg = config['models']['cnn']
    train_cfg = config['training']
    # Option: AE input type
    ae_input_type = cnn_cfg.get('ae_input_type', 'features')  # 'features' or 'features+embeddings'
    # Data loading
    train_df = pd.read_csv("data/processed/train_features.csv")
    test_df = pd.read_csv("data/processed/test_features.csv")
    import ast
    train_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in train_df['input_ids']]
    test_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in test_df['input_ids']]
    max_len = cnn_cfg['max_seq_length']
    train_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in train_input_ids]
    test_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in test_input_ids]
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
    # Standardize
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    X_test_features = scaler.transform(X_test_features)
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
    # Train/val split
    X_train_ids, X_val_ids, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
        train_input_ids, X_features, y,
        test_size=train_cfg['validation_split'],
        random_state=train_cfg['seed']
    )
    # AE config
    ae_hidden = cnn_cfg.get('hidden_dim', 192)
    ae_latent = cnn_cfg.get('ae_latent_dim', max(32, ae_hidden//2))
    ae_dropout = cnn_cfg.get('dropout', 0.3)
    ae_epochs = cnn_cfg.get('ae_epochs', 30)
    ae_lr = cnn_cfg.get('ae_lr', 1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Train AE
    print(f"\nTraining feature autoencoder on: {ae_input_type}")
    ae = FeatureAutoencoder(input_dim=X_train_feat.shape[1], hidden_dim=ae_hidden, latent_dim=ae_latent, dropout=ae_dropout)
    ae_loader = DataLoader(FeatureOnlyDataset(X_train_feat), batch_size=64, shuffle=True)
    ae = train_autoencoder(ae, ae_loader, ae_epochs, ae_lr, device)
    # Extract latent features
    ae.eval()
    with torch.no_grad():
        train_latent = ae.encode(torch.FloatTensor(X_train_feat).to(device)).cpu().numpy()
        val_latent = ae.encode(torch.FloatTensor(X_val_feat).to(device)).cpu().numpy()
        test_latent = ae.encode(torch.FloatTensor(X_test_features).to(device)).cpu().numpy()
    # Print AE input/output/latent dimensions and a sample of latent features
    print(f"\nAE input dim: {ae.encoder[0].in_features}")
    print(f"AE latent dim: {ae.encoder[-2].out_features}")
    print(f"AE output dim: {ae.decoder[-1].out_features}")
    print(f"Latent feature array shape: {train_latent.shape}")
    print("Sample AE latent features (first 3 samples, first 8 dims):")
    print(np.round(train_latent[:3, :8], 4))
    # If ae_input_type == 'features', prepare latent for concat to each token embedding
    if ae_input_type == 'features':
        # For each sample, repeat latent vector for each token (seq_len)
        seq_len = max_len
        train_latent = np.stack([np.tile(lat, (seq_len, 1)) for lat in train_latent])
        val_latent = np.stack([np.tile(lat, (seq_len, 1)) for lat in val_latent])
        test_latent = np.stack([np.tile(lat, (seq_len, 1)) for lat in test_latent])
    # CNN config
    max_token_id = int(max(max(seq) for seq in X_train_ids))
    vocab_size = max_token_id + 1
    cnn = CNNModel(
        vocab_size=vocab_size,
        embedding_dim=cnn_cfg['embedding_dim'],
        num_filters=cnn_cfg.get('num_filters', 64),
        kernel_sizes=cnn_cfg.get('kernel_sizes', [3,4,5]),
        dropout=cnn_cfg['dropout'],
        latent_dim=ae_latent
    ).to(device)
    # Datasets
    train_dataset = CNNWithLatentDataset(X_train_ids, train_latent, y_train)
    val_dataset = CNNWithLatentDataset(X_val_ids, val_latent, y_val)
    test_dataset = CNNWithLatentDataset(test_input_ids, test_latent)
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
            'use_scheduler': train_cfg.get('use_scheduler', False)
        }
        mlflow.log_params(params)
        # Loss, optimizer, scheduler
        loss_fn = train_cfg.get('loss_function', 'MSELoss')
        criterion = nn.L1Loss() if loss_fn == 'L1Loss' else nn.MSELoss()
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
            for batch_ids, batch_latent, batch_labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
                batch_ids = batch_ids.to(device)
                batch_latent = batch_latent.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                outputs = cnn(batch_ids, batch_latent)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds.extend(outputs.detach().cpu().numpy())
                targets.extend(batch_labels.cpu().numpy())
            train_mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
            cnn.eval()
            val_loss, val_mae = 0, 0
            vpreds, vtargets = [], []
            with torch.no_grad():
                for batch_ids, batch_latent, batch_labels in tqdm(val_loader, desc=f"Val {epoch+1}"):
                    batch_ids = batch_ids.to(device)
                    batch_latent = batch_latent.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = cnn(batch_ids, batch_latent)
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
            for batch_ids, batch_latent in tqdm(test_loader, desc="Predict"):
                batch_ids = batch_ids.to(device)
                batch_latent = batch_latent.to(device)
                outputs = cnn(batch_ids, batch_latent)
                preds.extend(outputs.cpu().numpy())
        preds = np.clip(preds, 0, 1)
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame({'example_id': test_ids, 'label': preds})
        output_path = output_dir / f"test-{model_name}.csv"
        output_df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved: {output_path}")
        mlflow.log_artifact(str(output_path))
        mlflow.log_metric('best_val_mae', best_val_mae)
        print(f"\n{'='*60}")
        print(f"✅ AE+CNN Training Complete!")
        print(f"Best Val MAE: {best_val_mae:.4f}")
        print(f"Model: models/{model_name}_best.pth")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
