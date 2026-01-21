"""Test data loading for RQ1 datasets."""
from src.data.configs import DATASET_CONFIGS

print("Testing HH-RLHF subsets...")
hh_helpful = DATASET_CONFIGS['Anthropic/hh-rlhf-helpful'](sanity_check=True)
hh_harmless = DATASET_CONFIGS['Anthropic/hh-rlhf-harmless'](sanity_check=True)
train_h = hh_helpful.get_preference_dataset('train')
train_harm = hh_harmless.get_preference_dataset('train')
print(f'HH Helpful: {len(train_h)} train samples')
print(f'HH Harmless: {len(train_harm)} train samples')
print('Sample keys:', list(train_h[0].keys()))

print("\nTesting UltraFeedback dimensions...")
uf_help = DATASET_CONFIGS['OpenBMB/UltraFeedback-helpfulness'](sanity_check=True)
uf_honesty = DATASET_CONFIGS['OpenBMB/UltraFeedback-honesty'](sanity_check=True)
train_uf_h = uf_help.get_preference_dataset('train')
train_uf_hon = uf_honesty.get_preference_dataset('train')
print(f'UF Helpfulness: {len(train_uf_h)} train samples')
print(f'UF Honesty: {len(train_uf_hon)} train samples')
print('Sample keys:', list(train_uf_h[0].keys()))

print("\nAll data loading tests passed!")
