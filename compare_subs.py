import pandas as pd
import numpy as np

v4  = pd.read_csv('d:/wids/submission_v4.csv')
# cv1 = pd.read_csv('d:/wids/submission_claude_v1.csv')
cv1 = pd.read_csv('d:/wids/submission_claude_v1_imputed_data.csv')

cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

print('=== MEAN PROBABILITIES ===')
for c in cols:
    diff = cv1[c].mean() - v4[c].mean()
    print(f'{c}: v4={v4[c].mean():.4f}  claude_v1={cv1[c].mean():.4f}  diff={diff:+.4f}')

print('\n=== CORRELATION (ranking agreement) ===')
for c in cols:
    print(f'{c}: {v4[c].corr(cv1[c]):.4f}')

print('\n=== PERCENTILE DISTRIBUTION ===')
print(f'{"col":<12} {"v4_25":>7} {"v4_50":>7} {"v4_75":>7} {"v4_max":>7} | {"cv1_25":>7} {"cv1_50":>7} {"cv1_75":>7} {"cv1_max":>7}')
for c in cols:
    v = v4[c].describe()
    k = cv1[c].describe()
    print(f'{c:<12} {v["25%"]:>7.4f} {v["50%"]:>7.4f} {v["75%"]:>7.4f} {v["max"]:>7.4f} | {k["25%"]:>7.4f} {k["50%"]:>7.4f} {k["75%"]:>7.4f} {k["max"]:>7.4f}')

print('\n=== FIRES WHERE MODELS STRONGLY DISAGREE (prob_72h diff > 0.20) ===')
merged = v4.merge(cv1, on='event_id', suffixes=('_v4','_cv1'))
big_diff = merged[abs(merged['prob_72h_v4'] - merged['prob_72h_cv1']) > 0.20].copy()
big_diff['delta'] = big_diff['prob_72h_cv1'] - big_diff['prob_72h_v4']
print(f'Count: {len(big_diff)}')
print(big_diff[['event_id','prob_72h_v4','prob_72h_cv1','delta']].sort_values('delta').to_string(index=False))
