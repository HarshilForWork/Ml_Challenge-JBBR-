"""
Hybrid Model Training Script
Combines CNN and LSTM architectures
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
    """Dataset for engineered features"""
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class HybridModel(nn.Module):
    """Hybrid CNN-LSTM model for regression"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(HybridModel, self).__init__()
        
        # CNN pathway
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # LSTM pathway
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # CNN path
        x_cnn = x.unsqueeze(1)
        x_cnn = self.conv_layers(x_cnn)
        
        # Global pooling
        cnn_avg = torch.mean(x_cnn, dim=2)
        cnn_max = torch.max(x_cnn, dim=2)[0]
        cnn_features = torch.cat([cnn_avg, cnn_max], dim=1)
        
        # LSTM path
        x_lstm = x_cnn.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm)
        lstm_features = lstm_out[:, -1, :]
        
        # Fusion
        combined = torch.cat([cnn_features, lstm_features], dim=1)
        output = self.fusion(combined)
        
        return output.squeeze()


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
    
    for batch_features, batch_labels in tqdm(dataloader, desc="Training"):
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
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
        for batch_features, batch_labels in tqdm(dataloader, desc="Validation"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
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
        for batch_features in tqdm(dataloader, desc="Predicting"):
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = "hybrid"
    model_config = config['models'][model_name]
    train_config = config['training']
    
    print(f"\n{'='*60}")
    print(f"Training Hybrid CNN-LSTM Model")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading data...")
    train_df = pd.read_csv("data/processed/train_features.csv")
    test_df = pd.read_csv("data/processed/test_features.csv")
    
    # Load feature correlations and select top features
    corr_df = pd.read_csv("data/processed/feature_correlations.csv", index_col=0)
    top_n = model_config.get('top_features', 30)
    top_features = corr_df.nlargest(top_n, 'correlation').index.tolist()
    
    print(f"Using top {len(top_features)} features based on correlation")
    print(f"Top 5 features: {top_features[:5]}")
    
    # Filter to only include top features that exist in the data
    available_features = [f for f in top_features if f in train_df.columns]
    feature_cols = available_features
    
    X = train_df[feature_cols].values
    y = train_df['label'].values
    X_test = test_df[feature_cols].values
    test_ids = test_df['example_id'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=train_config['validation_split'],
        random_state=train_config['seed']
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    train_dataset = FeatureDataset(X_train, y_train)
    val_dataset = FeatureDataset(X_val, y_val)
    test_dataset = FeatureDataset(X_test)
    
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
            'input_dim': X_train.shape[1],
            'hidden_dim': model_config['hidden_dim'],
            'num_layers': model_config['num_layers'],
            'dropout': model_config['dropout'],
            'batch_size': model_config['batch_size'],
            'learning_rate': model_config['learning_rate'],
            'num_epochs': model_config['num_epochs'],
            'top_features': len(feature_cols)
        }
        mlflow.log_params(params)
        
        model = HybridModel(
            input_dim=X_train.shape[1],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
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
