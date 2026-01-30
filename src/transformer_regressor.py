import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yaml
import mlflow
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ast
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from catboost import CatBoostRegressor

# Load environment variables
load_dotenv()

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert string values to proper types
    transformer_config = config['transformer_regressor']
    
    # Handle learning rate scientific notation
    if isinstance(transformer_config['learning_rate'], str):
        transformer_config['learning_rate'] = float(transformer_config['learning_rate'])
    
    return transformer_config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=512):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerRegressor(nn.Module):
    """Transformer encoder with optional feature concatenation - matches transformer.py"""
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=4, num_heads=8, 
                 max_seq_length=512, dropout=0.4, use_engineered_features=True, num_features=40):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_engineered_features = use_engineered_features
        
        # Token embedding (matches transformer.py)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer encoder blocks (matches transformer.py)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # matches transformer.py
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Determine FC input dimension
        fc_input_dim = hidden_dim
        if use_engineered_features:
            fc_input_dim += num_features
        
        # FC layers (matches transformer.py but without sigmoid for regression)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.num_features = num_features
    
    def forward(self, input_ids, attention_mask=None, features=None, return_embeddings=False):
        # Clip token IDs to valid range (matches transformer.py)
        input_ids = torch.clamp(input_ids, 0, self.embedding.num_embeddings - 1)
        
        # Token embeddings + projection (matches transformer.py)
        x = self.embedding(input_ids)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Transformer encoding (matches transformer.py - no attention mask used)
        x = self.transformer(x)
        
        # Global average pooling (matches transformer.py)
        pooled = x.mean(dim=1)
        
        # Return embeddings if requested (for CatBoost training)
        if return_embeddings:
            return pooled
        
        # Concatenate with engineered features if available
        if self.use_engineered_features and features is not None:
            combined = torch.cat([pooled, features], dim=1)
        else:
            combined = pooled
        
        # Final prediction (no sigmoid for regression)
        logits = self.fc(combined)
        
        return logits.squeeze(-1)


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for regression"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.SmoothL1Loss()
    
    def forward(self, pred, target):
        # Add noise to targets for label smoothing
        if self.training:
            noise = torch.randn_like(target) * self.smoothing * 0.1
            target = torch.clamp(target + noise, 0, 1)
        return self.criterion(pred, target)


class QualityDataset(Dataset):
    """Dataset for quality regression training"""
    
    def __init__(self, input_ids, attention_masks, features, labels, max_length=256):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.features = features
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx], 
            'features': self.features[idx],
            'labels': self.labels[idx]
        }
        
        # Data augmentation: add small noise to features during training
        if hasattr(self, 'training') and self.training:
            noise = torch.randn_like(item['features']) * 0.01
            item['features'] = item['features'] + noise
            
        return item


def load_and_prepare_data(config):
    """Load and prepare training data"""
    print("Loading training data...")
    train_df = pd.read_csv('data/processed/train_features.csv')
    
    print("Processing tokenized data...")
    input_ids_list = []
    attention_masks_list = []
    max_seq_length = config['max_seq_length']
    
    for _, row in tqdm(train_df.iterrows(), desc="Loading data", total=len(train_df)):
        # Parse string representations back to lists
        input_ids = ast.literal_eval(row['input_ids'])
        attention_mask = ast.literal_eval(row['attention_mask'])
        
        # Pad or truncate to max_length
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            padding_length = max_seq_length - len(input_ids)
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
    
    # Convert to tensors using NumPy for efficiency
    input_ids = torch.from_numpy(np.array(input_ids_list, dtype=np.int64))
    attention_masks = torch.from_numpy(np.array(attention_masks_list, dtype=np.int64))
    
    # Extract engineered features (top 40 features like transformer.py)
    # Load feature correlations and get top features
    try:
        corr_df = pd.read_csv("data/processed/feature_correlations.csv", index_col=0)
        top_features = corr_df.nlargest(config['num_features'], 'correlation').index.tolist()
        available_features = [f for f in top_features if f in train_df.columns]
        print(f"Using top {len(available_features)} engineered features (like transformer.py)")
        print(f"Top 5 features: {available_features[:5]}")
    except FileNotFoundError:
        # Fallback to original 26 features if correlation file not found
        available_features = [
            'seq_length', 'padding_ratio', 'log_length', 'unique_tokens', 'type_token_ratio',
            'unique_ratio', 'token_mean', 'token_std', 'token_median', 'token_min',
            'token_max', 'token_range', 'token_skewness', 'token_kurtosis', 'max_token_freq',
            'entropy', 'gini_coefficient', 'repetition_rate', 'first_token', 'last_token',
            'first_last_same', 'avg_token_distance', 'unique_bigrams', 'bigram_diversity',
            'unique_trigrams', 'trigram_diversity'
        ]
        print(f"Using fallback {len(available_features)} engineered features")
    
    # Convert to NumPy for efficient processing
    features_np = train_df[available_features].values.astype(np.float32)
    labels_np = train_df['label'].values.astype(np.float32)
    
    # Normalize engineered features using NumPy
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_np).astype(np.float32)
    
    # Convert to tensors using NumPy
    features = torch.from_numpy(features_normalized)
    labels = torch.from_numpy(labels_np)
    
    print(f"Data shapes:")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Attention masks: {attention_masks.shape}")
    print(f"  Features: {features.shape} (normalized)")
    print(f"  Labels: {labels.shape}")
    print(f"  Label range: [{labels.min():.3f}, {labels.max():.3f}]")
    
    return input_ids, attention_masks, features, labels, scaler


def extract_embeddings(model, input_ids, attention_masks, batch_size, device):
    """Extract transformer embeddings from data"""
    model.eval()
    embeddings = []
    
    # Create dataset for embedding extraction
    dataset = QualityDataset(input_ids, attention_masks, 
                           torch.zeros((len(input_ids), 26)), torch.zeros(len(input_ids)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            
            # Extract embeddings (not predictions)
            batch_embeddings = model(input_ids_batch, attention_mask_batch, return_embeddings=True)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def train_model(config):
    """Train the transformer regressor"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    input_ids, attention_masks, features, labels, scaler = load_and_prepare_data(config)
    
    # Train/validation split
    print("Creating train/validation split...")
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=None
    )
    
    # Create datasets
    train_dataset = QualityDataset(
        input_ids[train_idx], attention_masks[train_idx], 
        features[train_idx], labels[train_idx], config['max_seq_length']
    )
    
    val_dataset = QualityDataset(
        input_ids[val_idx], attention_masks[val_idx],
        features[val_idx], labels[val_idx], config['max_seq_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Get vocab size from data
    vocab_size = max([max(ids) for ids in input_ids]) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = TransformerRegressor(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout'],
        use_engineered_features=config['use_engineered_features'],
        num_features=config['num_features']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function - use L1Loss (MAE)
    criterion = nn.L1Loss()
    print("Using L1Loss (MAE) for better regression performance")
    
    # Optimizer selection
    optimizer_name = config.get('optimizer', 'adamw').lower()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            momentum=0.9
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"Using optimizer: {optimizer_name.upper()}")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler_factor'],
        patience=config['scheduler_patience'], verbose=True
    )
    
    # Training loop
    best_val_mae = float('inf')  # Track MAE instead of loss
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        num_batches = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for batch in train_bar:
            optimizer.zero_grad()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            # Apply mixup augmentation if enabled (disabled in current config)
            # if config.get('mixup_alpha', 0) > 0:
            #     mixed_features, y_a, y_b, lam = mixup_data(features, labels, config['mixup_alpha'])
            #     features = mixed_features
            
            # Forward pass
            predictions = model(input_ids, attention_mask, features)
            predictions = torch.clamp(predictions, 0, 1)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.get('gradient_clipping', True):
                clip_value = config.get('max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(predictions - labels)).item()
            num_batches += 1
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{train_mae/num_batches:.4f}'
            })
        
        avg_train_loss = train_loss / num_batches
        avg_train_mae = train_mae / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            
            for batch in val_bar:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                predictions = model(input_ids, attention_mask, features)
                predictions = torch.clamp(predictions, 0, 1)
                
                # Calculate loss
                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(predictions - labels)).item()
                
                # Store for metrics
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        # Calculate additional metrics
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to MLflow
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("train_mae", avg_train_mae, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_mae", avg_val_mae, step=epoch)
        mlflow.log_metric("val_rmse", val_rmse, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
        
        # Learning rate scheduling - use MAE instead of loss
        scheduler.step(avg_val_mae)
        
        # Early stopping and model saving - based on MAE
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            
            # Save best model
            model_path = 'models/transformer_regressor_best.pth'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'vocab_size': vocab_size,
                'epoch': epoch,
                'best_val_mae': best_val_mae,  # Save MAE instead of loss
                'scaler': scaler
            }, model_path)
            print(f"  Saved best model with val_MAE: {best_val_mae:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping after {epoch+1} epochs (no improvement in MAE)")
            break
    
    return model, vocab_size, scaler


def train_catboost_with_embeddings(config):
    """Train CatBoost on engineered features + transformer embeddings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    input_ids, attention_masks, features, labels, scaler = load_and_prepare_data(config)
    
    # Load trained transformer model
    print("Loading trained transformer model...")
    model_path = 'models/transformer_regressor_best.pth'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab_size = checkpoint['vocab_size']
    scaler = checkpoint['scaler']
    
    # Initialize model
    model = TransformerRegressor(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout'],
        use_engineered_features=config['use_engineered_features'],
        num_features=config['num_features']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Extracting transformer embeddings...")
    transformer_embeddings = extract_embeddings(model, input_ids, attention_masks, 
                                               config['batch_size'], device)
    
    print(f"Transformer embeddings shape: {transformer_embeddings.shape}")
    
    # Combine engineered features + transformer embeddings
    combined_features = np.hstack([features.numpy(), transformer_embeddings])
    labels_np = labels.numpy()
    
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Features breakdown: {features.shape[1]} engineered + {transformer_embeddings.shape[1]} embeddings")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        combined_features, labels_np, test_size=0.2, random_state=42
    )
    
    print(f"Training CatBoost on {X_train.shape[0]} samples...")
    
    # Train CatBoost
    catboost_model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        loss_function='MAE',
        eval_metric='MAE',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100
    )
    
    catboost_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        plot=False
    )
    
    # Validation predictions
    val_predictions = catboost_model.predict(X_val)
    val_predictions = np.clip(val_predictions, 0, 1)
    
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"\nCatBoost Validation Results:")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    
    # Log CatBoost metrics
    mlflow.log_metric("catboost_val_mae", val_mae)
    mlflow.log_metric("catboost_val_rmse", val_rmse)
    
    # Save CatBoost model
    catboost_path = 'models/catboost_transformer_embeddings.cbm'
    os.makedirs(os.path.dirname(catboost_path), exist_ok=True)
    catboost_model.save_model(catboost_path)
    
    return catboost_model, model


def predict_and_submit(model, vocab_size, scaler, config):
    """Generate predictions and create submission file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    print("Loading test data...")
    test_df = pd.read_csv('data/processed/test_features.csv')
    
    # Process test data
    input_ids_list = []
    attention_masks_list = []
    max_seq_length = config['max_seq_length']
    
    for _, row in tqdm(test_df.iterrows(), desc="Processing test data", total=len(test_df)):
        input_ids = ast.literal_eval(row['input_ids'])
        attention_mask = ast.literal_eval(row['attention_mask'])
        
        # Pad or truncate
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            padding_length = max_seq_length - len(input_ids)
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
    
    # Convert to tensors using NumPy for efficiency
    input_ids = torch.from_numpy(np.array(input_ids_list, dtype=np.int64))
    attention_masks = torch.from_numpy(np.array(attention_masks_list, dtype=np.int64))
    
    # Extract features - use same logic as training
    try:
        corr_df = pd.read_csv("data/processed/feature_correlations.csv", index_col=0)
        top_features = corr_df.nlargest(config['num_features'], 'correlation').index.tolist()
        available_features = [f for f in top_features if f in test_df.columns]
    except FileNotFoundError:
        # Fallback to original features if correlation file not found
        available_features = [
            'seq_length', 'padding_ratio', 'log_length', 'unique_tokens', 'type_token_ratio',
            'unique_ratio', 'token_mean', 'token_std', 'token_median', 'token_min',
            'token_max', 'token_range', 'token_skewness', 'token_kurtosis', 'max_token_freq',
            'entropy', 'gini_coefficient', 'repetition_rate', 'first_token', 'last_token',
            'first_last_same', 'avg_token_distance', 'unique_bigrams', 'bigram_diversity',
            'unique_trigrams', 'trigram_diversity'
        ]
    # Convert to NumPy for efficient processing
    features_np = test_df[available_features].values.astype(np.float32)
    
    # Normalize test features using training scaler
    features_normalized = scaler.transform(features_np).astype(np.float32)
    features = torch.from_numpy(features_normalized)
    
    # Create test dataset and loader
    test_dataset = QualityDataset(input_ids, attention_masks, features, 
                                torch.zeros(len(test_df)), config['max_seq_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate predictions
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            
            # Predict
            preds = model(input_ids, attention_mask, features)
            preds = torch.clamp(preds, 0, 1)  # Ensure valid range
            
            predictions.extend(preds.cpu().numpy())
    
    # Create submission
    predictions = np.array(predictions)
    predictions = np.clip(predictions, 0, 1)  # Double-check clipping
    
    submission = pd.DataFrame({
        'example_id': test_df['example_id'].values,
        'label': predictions
    })
    
    # Save submission
    submission_path = 'data/output/transformer_regressor_submission.csv'
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission saved to {submission_path}")
    print(f"Prediction statistics:")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    
    return submission_path


def predict_catboost_with_embeddings(transformer_model, catboost_model, config):
    """Generate CatBoost predictions using transformer embeddings + engineered features"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading test data for CatBoost prediction...")
    test_df = pd.read_csv('data/processed/test_features.csv')
    
    # Process test data
    input_ids_list = []
    attention_masks_list = []
    max_seq_length = config['max_seq_length']
    
    for _, row in tqdm(test_df.iterrows(), desc="Processing test data", total=len(test_df)):
        input_ids = ast.literal_eval(row['input_ids'])
        attention_mask = ast.literal_eval(row['attention_mask'])
        
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            padding_length = max_seq_length - len(input_ids)
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
    
    # Convert to tensors using NumPy for efficiency
    input_ids = torch.from_numpy(np.array(input_ids_list, dtype=np.int64))
    attention_masks = torch.from_numpy(np.array(attention_masks_list, dtype=np.int64))
    
    # Extract engineered features - use same logic as training
    try:
        corr_df = pd.read_csv("data/processed/feature_correlations.csv", index_col=0)
        top_features = corr_df.nlargest(40, 'correlation').index.tolist()  # Use 40 from config
        available_features = [f for f in top_features if f in test_df.columns]
    except FileNotFoundError:
        # Fallback to original features if correlation file not found
        available_features = [
            'seq_length', 'padding_ratio', 'log_length', 'unique_tokens', 'type_token_ratio',
            'unique_ratio', 'token_mean', 'token_std', 'token_median', 'token_min',
            'token_max', 'token_range', 'token_skewness', 'token_kurtosis', 'max_token_freq',
            'entropy', 'gini_coefficient', 'repetition_rate', 'first_token', 'last_token',
            'first_last_same', 'avg_token_distance', 'unique_bigrams', 'bigram_diversity',
            'unique_trigrams', 'trigram_diversity'
        ]
    # Use NumPy for efficient processing
    engineered_features = test_df[available_features].values.astype(np.float32)
    
    # Load scaler from checkpoint
    model_path = 'models/transformer_regressor_best.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    scaler = checkpoint['scaler']
    
    # Normalize test features
    engineered_features = scaler.transform(engineered_features)
    
    # Extract transformer embeddings
    print("Extracting test embeddings...")
    transformer_embeddings = extract_embeddings(transformer_model, input_ids, attention_masks,
                                               config['batch_size'], device)
    
    # Combine features
    combined_features = np.hstack([engineered_features, transformer_embeddings])
    print(f"Combined test features shape: {combined_features.shape}")
    
    # Predict with CatBoost
    predictions = catboost_model.predict(combined_features)
    predictions = np.clip(predictions, 0, 1)
    
    # Create submission
    submission = pd.DataFrame({
        'example_id': test_df['example_id'].values,
        'label': predictions
    })
    
    submission_path = 'data/output/catboost_transformer_submission.csv'
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)
    
    print(f"\nCatBoost submission saved to {submission_path}")
    print(f"Prediction statistics:")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    
    return submission_path


def main():
    """Main training and prediction pipeline"""
    
    # Load configuration
    config = load_config()
    print("Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup MLflow
    mlflow.set_experiment("transformer_regressor")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Train transformer model
        print("\nStarting transformer regressor training...")
        model, vocab_size, scaler = train_model(config)
        
        # Load best transformer model
        model_path = 'models/transformer_regressor_best.pth'
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Train CatBoost with embeddings + engineered features
        print("\nTraining CatBoost with transformer embeddings...")
        catboost_model, transformer_model = train_catboost_with_embeddings(config)
        
        # Generate CatBoost predictions
        catboost_submission_path = predict_catboost_with_embeddings(transformer_model, catboost_model, config)
        
        # Also generate direct transformer predictions for comparison
        transformer_submission_path = predict_and_submit(model, vocab_size, scaler, config)
        
        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact('models/catboost_transformer_embeddings.cbm')
        mlflow.log_artifact(catboost_submission_path)
        mlflow.log_artifact(transformer_submission_path)
        
        print("\nTraining and prediction complete!")
        print(f"Transformer submission: {transformer_submission_path}")
        print(f"CatBoost submission: {catboost_submission_path}")


if __name__ == "__main__":
    main()