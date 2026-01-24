"""
Feature Engineering for Prompt Quality Prediction

Since token IDs are obfuscated, we extract statistical and structural features
that might correlate with prompt quality.
"""
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from typing import List, Dict


class FeatureEngineer:
    """Extract features from obfuscated token sequences"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, input_ids: List[int], attention_mask: List[int]) -> Dict[str, float]:
        """Extract all features from a single example"""
        features = {}
        
        # Get actual tokens (non-padded)
        actual_tokens = [tid for tid, mask in zip(input_ids, attention_mask) if mask == 1]
        
        # 1. LENGTH FEATURES
        features.update(self._length_features(actual_tokens, input_ids))
        
        # 2. DIVERSITY FEATURES
        features.update(self._diversity_features(actual_tokens))
        
        # 3. STATISTICAL FEATURES
        features.update(self._statistical_features(actual_tokens))
        
        # 4. REPETITION FEATURES
        features.update(self._repetition_features(actual_tokens))
        
        # 5. POSITIONAL FEATURES
        features.update(self._positional_features(actual_tokens, input_ids))
        
        # 6. N-GRAM FEATURES
        features.update(self._ngram_features(actual_tokens))
        
        # 7. SPECIAL TOKEN FEATURES
        features.update(self._special_token_features(actual_tokens))
        
        return features
    
    def _length_features(self, tokens: List[int], full_seq: List[int]) -> Dict[str, float]:
        """Features related to sequence length"""
        return {
            'seq_length': len(tokens),
            'padding_ratio': 1 - (len(tokens) / len(full_seq)) if len(full_seq) > 0 else 0,
            'log_length': np.log1p(len(tokens))
        }
    
    def _diversity_features(self, tokens: List[int]) -> Dict[str, float]:
        """Features related to token diversity"""
        if len(tokens) == 0:
            return {
                'unique_tokens': 0,
                'type_token_ratio': 0,
                'unique_ratio': 0
            }
        
        unique = len(set(tokens))
        return {
            'unique_tokens': unique,
            'type_token_ratio': unique / len(tokens),  # Vocabulary richness
            'unique_ratio': unique / len(tokens)
        }
    
    def _statistical_features(self, tokens: List[int]) -> Dict[str, float]:
        """Statistical features of token IDs"""
        if len(tokens) == 0:
            return {
                'token_mean': 0,
                'token_std': 0,
                'token_median': 0,
                'token_min': 0,
                'token_max': 0,
                'token_range': 0,
                'token_skewness': 0,
                'token_kurtosis': 0
            }
        
        tokens_arr = np.array(tokens)
        return {
            'token_mean': float(np.mean(tokens_arr)),
            'token_std': float(np.std(tokens_arr)),
            'token_median': float(np.median(tokens_arr)),
            'token_min': float(np.min(tokens_arr)),
            'token_max': float(np.max(tokens_arr)),
            'token_range': float(np.ptp(tokens_arr)),
            'token_skewness': float(stats.skew(tokens_arr)),
            'token_kurtosis': float(stats.kurtosis(tokens_arr))
        }
    
    def _repetition_features(self, tokens: List[int]) -> Dict[str, float]:
        """Features related to token repetition"""
        if len(tokens) == 0:
            return {
                'max_token_freq': 0,
                'entropy': 0,
                'gini_coefficient': 0,
                'repetition_rate': 0
            }
        
        freq = Counter(tokens)
        freqs = np.array(list(freq.values()))
        
        # Entropy (higher = more diverse)
        probs = freqs / len(tokens)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Gini coefficient (measure of inequality)
        sorted_freqs = np.sort(freqs)
        n = len(sorted_freqs)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_freqs)) / (n * np.sum(sorted_freqs)) - (n + 1) / n
        
        return {
            'max_token_freq': float(max(freqs)),
            'entropy': float(entropy),
            'gini_coefficient': float(gini),
            'repetition_rate': float(len(tokens) - len(freq)) / len(tokens)
        }
    
    def _positional_features(self, tokens: List[int], full_seq: List[int]) -> Dict[str, float]:
        """Features related to token positions"""
        if len(tokens) < 2:
            return {
                'first_token': tokens[0] if tokens else 0,
                'last_token': tokens[-1] if tokens else 0,
                'first_last_same': 0,
                'avg_token_distance': 0
            }
        
        # Consecutive token differences
        diffs = np.diff(tokens)
        
        return {
            'first_token': float(tokens[0]),
            'last_token': float(tokens[-1]),
            'first_last_same': float(tokens[0] == tokens[-1]),
            'avg_token_distance': float(np.mean(np.abs(diffs)))
        }
    
    def _ngram_features(self, tokens: List[int]) -> Dict[str, float]:
        """N-gram based features"""
        if len(tokens) < 2:
            return {
                'unique_bigrams': 0,
                'bigram_diversity': 0,
                'unique_trigrams': 0,
                'trigram_diversity': 0
            }
        
        # Bigrams
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        unique_bigrams = len(set(bigrams))
        
        # Trigrams
        trigrams = []
        if len(tokens) >= 3:
            trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
        unique_trigrams = len(set(trigrams))
        
        return {
            'unique_bigrams': unique_bigrams,
            'bigram_diversity': unique_bigrams / len(bigrams) if bigrams else 0,
            'unique_trigrams': unique_trigrams,
            'trigram_diversity': unique_trigrams / len(trigrams) if trigrams else 0
        }
    
    def _special_token_features(self, tokens: List[int]) -> Dict[str, float]:
        """Features for potential special tokens"""
        if len(tokens) == 0:
            return {
                'has_low_value_tokens': 0,
                'low_token_ratio': 0,
                'high_token_ratio': 0,
                'mid_range_ratio': 0
            }
        
        tokens_arr = np.array(tokens)
        
        # Assuming tokens < 100 might be special tokens
        low_tokens = np.sum(tokens_arr < 100)
        high_tokens = np.sum(tokens_arr > 10000)
        
        return {
            'has_low_value_tokens': float(low_tokens > 0),
            'low_token_ratio': float(low_tokens / len(tokens)),
            'high_token_ratio': float(high_tokens / len(tokens)),
            'mid_range_ratio': float(1 - (low_tokens + high_tokens) / len(tokens))
        }
    
    def process_dataset(self, data: List[Dict]) -> pd.DataFrame:
        """Process entire dataset and return feature DataFrame"""
        features_list = []
        
        for item in data:
            features = self.extract_features(
                item['input_ids'],
                item['attention_mask']
            )
            features['example_id'] = item['example_id']
            if 'label' in item:
                features['label'] = item['label']
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Store feature names (excluding example_id and label)
        self.feature_names = [col for col in df.columns 
                             if col not in ['example_id', 'label']]
        
        return df


def analyze_features(df: pd.DataFrame, label_col: str = 'label'):
    """Analyze feature importance and correlations"""
    if label_col not in df.columns:
        print("No labels available for analysis")
        return
    
    feature_cols = [col for col in df.columns if col not in ['example_id', label_col]]
    
    # Calculate correlations with target
    correlations = df[feature_cols].corrwith(df[label_col]).abs().sort_values(ascending=False)
    
    print("\n" + "="*60)
    print("Top 10 Features by Correlation with Label")
    print("="*60)
    for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"{i:2d}. {feat:30s} {corr:.4f}")
    
    print("\n" + "="*60)
    print("Feature Statistics")
    print("="*60)
    print(df[feature_cols].describe())
    
    return correlations


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import ast
    
    print("="*60)
    print("Feature Engineering Pipeline")
    print("="*60)
    
    # Load CSV data
    print("\nLoading training data from CSV...")
    train_df = pd.read_csv("input_data/train.csv")
    
    # Convert string representations back to lists
    print("Parsing token sequences...")
    train_data = []
    for _, row in train_df.iterrows():
        train_data.append({
            'example_id': row['example_id'],
            'input_ids': ast.literal_eval(row['input_ids']),
            'attention_mask': ast.literal_eval(row['attention_mask']),
            'label': row['label']
        })
    
    print(f"Loaded {len(train_data)} training examples")
    
    # Extract features
    print("\nExtracting features...")
    engineer = FeatureEngineer()
    train_features = engineer.process_dataset(train_data)
    
    print(f"\nExtracted {len(engineer.feature_names)} features:")
    for feat in engineer.feature_names:
        print(f"  - {feat}")
    
    # Save processed features
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "train_features.csv"
    train_features.to_csv(output_path, index=False)
    print(f"\n✓ Saved features to: {output_path}")
    
    # Analyze features
    print("\nAnalyzing features...")
    correlations = analyze_features(train_features)
    
    # Save correlation analysis
    corr_path = output_dir / "feature_correlations.csv"
    correlations.to_csv(corr_path, header=['correlation'])
    print(f"✓ Saved correlations to: {corr_path}")
    
    print(f"\n✓ Feature DataFrame shape: {train_features.shape}")
    print("\nSample features:")
    print(train_features.head())
    
    # Process test data
    print("\n" + "="*60)
    print("Processing test data...")
    print("="*60)
    
    test_df = pd.read_csv("input_data/test.csv")
    test_data = []
    for _, row in test_df.iterrows():
        test_data.append({
            'example_id': row['example_id'],
            'input_ids': ast.literal_eval(row['input_ids']),
            'attention_mask': ast.literal_eval(row['attention_mask'])
        })
    
    test_features = engineer.process_dataset(test_data)
    test_output_path = output_dir / "test_features.csv"
    test_features.to_csv(test_output_path, index=False)
    print(f"✓ Saved test features to: {test_output_path}")
    print(f"✓ Test DataFrame shape: {test_features.shape}")
    
    print("\n" + "="*60)
    print("✅ Feature Engineering Complete!")
    print("="*60)
    print(f"\nProcessed files saved in: {output_dir}/")
    print("  - train_features.csv")
    print("  - test_features.csv")
    print("  - feature_correlations.csv")
    print("\nNext: Track with DVC and train models!")
