import pandas as pd
import ast

# Check training data
df = pd.read_csv('data/processed/train_features.csv')
max_ids = df['input_ids'].apply(ast.literal_eval).apply(max)
min_ids = df['input_ids'].apply(ast.literal_eval).apply(min)

print(f'Training data:')
print(f'  Max token ID: {max_ids.max()}')
print(f'  Min token ID: {min_ids.min()}')

# Check test data
df_test = pd.read_csv('data/processed/test_features.csv')
max_ids_test = df_test['input_ids'].apply(ast.literal_eval).apply(max)
min_ids_test = df_test['input_ids'].apply(ast.literal_eval).apply(min)

print(f'\nTest data:')
print(f'  Max token ID: {max_ids_test.max()}')
print(f'  Min token ID: {min_ids_test.min()}')

print(f'\nRequired vocab_size: {max(max_ids.max(), max_ids_test.max()) + 1}')
