"""
Transformer Model Training Script
Trains Transformer encoder on engineered features
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


class FeatureDataset(Dataset):
    """Dataset for token sequences and optional engineered features"""
    def __init__(self, input_ids, features=None, labels=None):
        self.input_ids = torch.LongTensor(input_ids)
        self.features = torch.FloatTensor(features) if features is not None else None
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            if self.features is not None:
                return self.input_ids[idx], self.features[idx], self.labels[idx]
            return self.input_ids[idx], self.labels[idx]
        if self.features is not None:
            return self.input_ids[idx], self.features[idx]
        return self.input_ids[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer encoder with optional feature concatenation"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, dropout, num_features=0):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        fc_input_dim = hidden_dim + num_features
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.num_features = num_features
    
    def forward(self, input_ids, features=None):
        # Clip token IDs to valid range [0, vocab_size-1]
        input_ids = torch.clamp(input_ids, 0, self.embedding.num_embeddings - 1)
        x = self.embedding(input_ids)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        
        if self.num_features > 0 and features is not None:
            x = torch.cat([x, features], dim=1)
        
        x = self.fc(x)
        return x.squeeze()


def plot_training_history(train_losses, val_losses, train_maes, val_maes, model_name, save_dir="graphs"):
    """Plot and save training history"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'{model_name.upper()} - Training Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_maes, 'b-', label='Train MAE', linewidth=2)
    ax2.plot(epochs, val_maes, 'r-', label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title(f'{model_name.upper()} - Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    best_epoch = np.argmin(val_maes) + 1
    best_mae = min(val_maes)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_epoch})')
    ax2.text(best_epoch, best_mae, f'  {best_mae:.4f}', fontsize=10, color='g')
    
    plt.tight_layout()
    graph_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return graph_path


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions, targets = [], []
    
    for batch_data in tqdm(dataloader, desc="Training"):
        if len(batch_data) == 3:
            batch_ids, batch_features, batch_labels = batch_data
            batch_ids, batch_features, batch_labels = batch_ids.to(device), batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_ids, batch_features)
        else:
            batch_ids, batch_labels = batch_data
            batch_ids, batch_labels = batch_ids.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_ids)
        
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(batch_labels.cpu().numpy())
    
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    return total_loss / len(dataloader), mae


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validation"):
            if len(batch_data) == 3:
                batch_ids, batch_features, batch_labels = batch_data
                batch_ids, batch_features, batch_labels = batch_ids.to(device), batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_ids, batch_features)
            else:
                batch_ids, batch_labels = batch_data
                batch_ids, batch_labels = batch_ids.to(device), batch_labels.to(device)
                outputs = model(batch_ids)
            
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_labels.cpu().numpy())
    
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    return total_loss / len(dataloader), mae


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Predicting"):
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                batch_ids, batch_features = batch_data
                batch_ids, batch_features = batch_ids.to(device), batch_features.to(device)
                outputs = model(batch_ids, batch_features)
            else:
                if isinstance(batch_data, (tuple, list)):
                    batch_ids = batch_data[0].to(device)
                else:
                    batch_ids = batch_data.to(device)
                outputs = model(batch_ids)
            
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = "transformer"
    model_config = config['models'][model_name]
    train_config = config['training']
    
    print(f"\n{'='*60}")
    print(f"Training Transformer Model")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading data...")
    train_df = pd.read_csv("data/processed/train_features.csv")
    test_df = pd.read_csv("data/processed/test_features.csv")
    
    # Parse input_ids from string
    import ast
    train_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in train_df['input_ids']]
    test_input_ids = [ast.literal_eval(ids) if isinstance(ids, str) else ids for ids in test_df['input_ids']]
    
    # Pad sequences to max_seq_length
    max_len = model_config['max_seq_length']
    train_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in train_input_ids]
    test_input_ids = [ids[:max_len] + [0]*(max_len-len(ids[:max_len])) for ids in test_input_ids]
    
    y = train_df['label'].values
    test_ids = test_df['example_id'].values
    
    # Load engineered features if top_features > 0
    top_n = model_config.get('top_features', 0)
    X_features = None
    X_test_features = None
    num_features = 0
    
    if top_n > 0:
        corr_df = pd.read_csv("data/processed/feature_correlations.csv", index_col=0)
        top_features = corr_df.nlargest(top_n, 'correlation').index.tolist()
        available_features = [f for f in top_features if f in train_df.columns]
        
        if available_features:
            print(f"Using top {len(available_features)} engineered features")
            print(f"Top 5 features: {available_features[:5]}")
            X_features = train_df[available_features].values
            X_test_features = test_df[available_features].values
            num_features = len(available_features)
        else:
            print("No valid engineered features found, using raw tokens only")
    else:
        print("Using raw input_ids only (no engineered features)")
    
    # Train/val split
    if X_features is not None:
        X_train_ids, X_val_ids, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
            train_input_ids, X_features, y,
            test_size=train_config['validation_split'],
            random_state=train_config['seed']
        )
    else:
        X_train_ids, X_val_ids, y_train, y_val = train_test_split(
            train_input_ids, y,
            test_size=train_config['validation_split'],
            random_state=train_config['seed']
        )
        X_train_feat = X_val_feat = None
    
    # Standardize engineered features if present
    scaler = None
    if X_train_feat is not None:
        scaler = StandardScaler()
        X_train_feat = scaler.fit_transform(X_train_feat)
        X_val_feat = scaler.transform(X_val_feat)
        X_test_features = scaler.transform(X_test_features)
    
    print(f"Train sequences: {len(X_train_ids)}, Val sequences: {len(X_val_ids)}, Test sequences: {len(test_input_ids)}")
    if num_features > 0:
        print(f"Feature dimensions: {num_features}")
    
    # Auto-detect vocab_size from training data only
    max_token_id = int(max(max(seq) for seq in X_train_ids))
    vocab_size = max_token_id + 1
    print(f"Auto-detected vocab_size: {vocab_size} (max training token ID: {max_token_id})")
    
    train_dataset = FeatureDataset(X_train_ids, X_train_feat, y_train)
    val_dataset = FeatureDataset(X_val_ids, X_val_feat, y_val)
    test_dataset = FeatureDataset(test_input_ids, X_test_features)
    
    train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'])
    
    # Setup MLflow with DagsHub or fallback to local
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', '')
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '')
        print(f"MLflow tracking: {tracking_uri}")
    else:
        print("MLflow tracking: Local (./mlruns)")
    
    # Set experiment (creates if doesn't exist, reuses if exists)
    experiment_name = f"prompt-quality-{model_name}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model_name}-run"):
        params = {
            'model': model_name,
            'vocab_size': vocab_size,
            'embedding_dim': model_config['embedding_dim'],
            'max_seq_length': model_config['max_seq_length'],
            'hidden_dim': model_config['hidden_dim'],
            'num_layers': model_config['num_layers'],
            'num_heads': model_config['num_heads'],
            'dropout': model_config['dropout'],
            'batch_size': model_config['batch_size'],
            'learning_rate': model_config['learning_rate'],
            'num_epochs': model_config['num_epochs'],
            'num_features': num_features,
            'top_features': top_n
        }
        mlflow.log_params(params)
        
        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            num_features=num_features
        ).to(device)
        
        print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
        
        best_val_mae = float('inf')
        patience_counter = 0
        
        train_losses_history = []
        val_losses_history = []
        train_maes_history = []
        val_maes_history = []
        
        for epoch in range(model_config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{model_config['num_epochs']}")
            
            train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_mae = validate(model, val_loader, criterion, device)
            
            train_losses_history.append(train_loss)
            val_losses_history.append(val_loss)
            train_maes_history.append(train_mae)
            val_maes_history.append(val_mae)
            
            print(f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
            
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_mae': val_mae
            }, step=epoch)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                
                model_path = f'models/{model_name}_best.pth'
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                print(f"✓ Best model saved! MAE: {best_val_mae:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= train_config['early_stopping_patience']:
                    print("\nEarly stopping triggered!")
                    break
        
        print("\nGenerating training plots...")
        graph_path = plot_training_history(
            train_losses_history, val_losses_history,
            train_maes_history, val_maes_history, model_name
        )
        print(f"✓ Training plot saved: {graph_path}")
        mlflow.log_artifact(graph_path)
        
        model.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
        
        print("\nGenerating predictions...")
        predictions = predict(model, test_loader, device)
        predictions = np.clip(predictions, 0, 1)
        
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_df = pd.DataFrame({
            'example_id': test_ids,
            'label': predictions
        })
        output_path = output_dir / f"test-{model_name}.csv"
        output_df.to_csv(output_path, index=False)
        
        print(f"✓ Predictions saved: {output_path}")
        mlflow.log_artifact(str(output_path))
        mlflow.log_metric('best_val_mae', best_val_mae)
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"Best Val MAE: {best_val_mae:.4f}")
        print(f"Model: models/{model_name}_best.pth")
        print(f"Output: {output_path}")
        print(f"Graph: {graph_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
