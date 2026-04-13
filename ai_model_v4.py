# ============================================================
# WiDS 2025 Datathon - Wildfire Survival Analysis (v4 - Advanced)
# ============================================================
# Upgrades from v3:
# 1. Implements a Two-Stage "Specialist/Generalist" training pipeline
#    based on data temporal resolution.
# 2. Replaces blending with Stacking using a Level 2 Meta-Model.
# 3. Includes advanced feature engineering and label artifact analysis.
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

# Models
import xgboost as xgb
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter

# Utility & Analysis
import os
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# ============================================================
# 0. CONFIGURATION & EXPERIMENT TRACKING
# ============================================================
USE_WANDB = False # SET TO True TO ENABLE LOGGING, False TO DISABLE
CONFIG = {
    "n_folds": 5,
    "n_repeats": 3, # Set to 1 for faster testing, 3 for robust final run
    "random_state": 42,
    "model_list": ['RSF', 'XGB'] # Add 'GBSA', 'LGB' for a full ensemble
}

if USE_WANDB:
    wandb.init(project="WiDS2025-Wildfire", name="v4_stacking_specialist", config=CONFIG)

# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================
print("=" * 70)
print("WiDS 2025 WILDFIRE SURVIVAL ANALYSIS (v4)")
print("=" * 70)

DATA_DIR = "." # Set your data directory here
# train_df = pd.read_csv(os.path.join(DATA_DIR, "train_fixed.csv"))
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_imputed_advanced.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_imputed_advanced.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

def engineer_features(df):
    """Create additional features from existing ones (v4)."""
    df = df.copy()
    
    # Core features
    df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    df['urgency_ratio'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    df['is_night'] = ((df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 6)).astype(int)
    df['is_summer'] = df['event_start_month'].isin([6, 7, 8]).astype(int)
    df['dist_risk_close'] = (df['dist_min_ci_0_5h'] < 3000).astype(int)
    df['has_growth'] = (df['area_growth_abs_0_5h'] > 0).astype(int)
    df['directional_threat'] = df['alignment_abs'] * df['closing_speed_abs_m_per_h']
    
    # Advanced features
    df['wind_driven_index'] = df['centroid_speed_m_per_h'] / (df['radial_growth_rate_m_per_h'] + 1)
    df['explosive_growth_index'] = df['area_growth_rel_0_5h'] / (np.log1p(df['area_first_ha']) + 0.1)
    df['night_growth_anomaly'] = df['is_night'] * df['area_growth_rate_ha_per_h']
    df['night_closing_anomaly'] = df['is_night'] * df['closing_speed_m_per_h']
    df['confidence_weighted_urgency'] = df['urgency_ratio'] * (1 - df['low_temporal_resolution_0_5h'])
    df['kinetic_threat_proxy'] = np.log1p(df['area_first_ha']) * (df['closing_speed_m_per_h']**2)
    
    return df

train_eng = engineer_features(train_df)
test_eng = engineer_features(test_df)

feature_cols = [c for c in train_eng.columns if c not in ['event_id', 'time_to_hit_hours', 'event'] and train_eng[c].dtype != 'object']
print(f"Features created: {len(feature_cols)}")

X_train = train_eng[feature_cols].values
y_time = train_eng['time_to_hit_hours'].values
y_event = train_eng['event'].values.astype(bool)
X_test = test_eng[feature_cols].values

y_surv = np.array(list(zip(y_event, y_time)), dtype=[('event', bool), ('time', float)])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

HORIZONS = [12, 24, 48, 72]

# ============================================================
# 2. LABEL ARTIFACT ANALYSIS (Inspired by Competitor)
# ============================================================
print("\n--- Investigating Label Artifacts ---")
hit_events = train_df[train_df['event'] == 1]
hit_events['time_fraction'] = hit_events['time_to_hit_hours'] % 1
plt.figure(figsize=(12, 5))
sns.histplot(hit_events['time_fraction'], bins=120)
plt.title('Distribution of the Fractional Part of Hit Times (e.g., 0.5 = 30 mins)')
plt.show()
print("Analysis: Look for spikes at 0.0, 0.25, 0.5, 0.75, which would indicate manual rounding.")

# ============================================================
# 3. TWO-STAGE "SPECIALIST/GENERALIST" MODEL TRAINING
# ============================================================
print("\n" + "=" * 70)
print("V4 MODEL TRAINING (Specialist/Generalist Approach)")
print("=" * 70)

# --- Define feature sets ---
unreliable_features = ['closing_speed', 'accel', 'slope', 'growth_rate', 'displacement', 'speed_m_per_h']
static_feature_cols = [c for c in feature_cols if not any(uf in c for uf in unreliable_features)]
static_feature_indices = [feature_cols.index(c) for c in static_feature_cols]

# --- Identify data row groups ---
is_low_res = train_eng['low_temporal_resolution_0_5h'] == 1
train_high_res_idx = np.where(~is_low_res)[0]
train_low_res_idx = np.where(is_low_res)[0]

# Initialize storage for final OOF and Test predictions from all models
oof_preds = {model: {'risk': np.zeros(len(X_train))} for model in CONFIG['model_list']}
test_preds = {model: {'risk': np.zeros(len(X_test))} for model in CONFIG['model_list']}

def train_specialized_model(train_indices, feature_indices, model_prefix):
    """Trains a full suite of models on a specific subset of data and returns their predictions."""
    print(f"\n--- Training {model_prefix} Model on {len(train_indices)} samples ---")
    
    oof_subset = {model: {'risk': np.zeros(len(train_indices))} for model in CONFIG['model_list']}
    test_subset = {model: {'risk': np.zeros(len(X_test))} for model in CONFIG['model_list']}
    
    subset_X = X_train_scaled[train_indices][:, feature_indices]
    subset_y_event = y_event[train_indices]
    
    subset_rskf = RepeatedStratifiedKFold(n_splits=CONFIG['n_folds'], n_repeats=CONFIG['n_repeats'], random_state=CONFIG['random_state'])
    total_subset_folds = CONFIG['n_folds'] * CONFIG['n_repeats']
    
    fold_num = 0
    for tr_sub_idx, val_sub_idx in subset_rskf.split(subset_X, subset_y_event):
        fold_num += 1
        print(f"  {model_prefix} Fold {fold_num}/{total_subset_folds}")
        
        original_tr_indices = train_indices[tr_sub_idx]
        original_val_indices = train_indices[val_sub_idx]
        
        X_tr = X_train_scaled[original_tr_indices][:, feature_indices]
        X_val = X_train_scaled[original_val_indices][:, feature_indices]
        y_tr = y_surv[original_tr_indices]

        # --- Train RSF ---
        if 'RSF' in CONFIG['model_list']:
            rsf_model = RandomSurvivalForest(random_state=CONFIG['random_state'] + fold_num, n_jobs=-1).fit(X_tr, y_tr)
            oof_subset['RSF']['risk'][val_sub_idx] += rsf_model.predict(X_val)
            test_subset['RSF']['risk'] += rsf_model.predict(X_test_scaled[:, feature_indices]) / total_subset_folds
        
        # --- Train XGBoost ---
        if 'XGB' in CONFIG['model_list']:
            y_tr_xgb = np.where(y_tr['event'], y_tr['time'], -y_tr['time'])
            xgb_model = xgb.XGBRegressor(random_state=CONFIG['random_state'] + fold_num, objective='survival:cox').fit(X_tr, y_tr_xgb)
            oof_subset['XGB']['risk'][val_sub_idx] += xgb_model.predict(X_val)
            test_subset['XGB']['risk'] += xgb_model.predict(X_test_scaled[:, feature_indices]) / total_subset_folds

    for model in CONFIG['model_list']:
        oof_subset[model]['risk'] /= CONFIG['n_repeats']

    return oof_subset, test_subset

# --- Run the two separate training pipelines ---
oof_high_res, test_high_res = train_specialized_model(train_high_res_idx, range(len(feature_cols)), "Specialist")
oof_low_res, test_low_res = train_specialized_model(train_low_res_idx, static_feature_indices, "Generalist")

# --- Combine results into the main prediction arrays ---
for model in CONFIG['model_list']:
    oof_preds[model]['risk'][train_high_res_idx] = oof_high_res[model]['risk']
    oof_preds[model]['risk'][train_low_res_idx] = oof_low_res[model]['risk']
    
    is_test_low_res = test_eng['low_temporal_resolution_0_5h'] == 1
    test_preds[model]['risk'] = np.where(is_test_low_res, test_low_res[model]['risk'], test_high_res[model]['risk'])
    
    # Log OOF score for this model
    c_idx_model, _, _, _, _ = concordance_index_censored(y_event, y_time, oof_preds[model]['risk'])
    print(f"{model} Overall OOF C-Index: {c_idx_model:.4f}")
    if USE_WANDB: wandb.log({f"oof_c_index_{model}": c_idx_model})

# ============================================================
# 4. STACKING (LEVEL 2 META-MODEL)
# ============================================================
print("\n" + "=" * 70)
print("STACKING (Training Level 2 Meta-Model)")
print("=" * 70)

# Create the training set for the meta-model from OOF predictions
meta_features_train = pd.DataFrame({
    f"{model}_risk_rank": rankdata(oof_preds[model]['risk'])
    for model in CONFIG['model_list']
})
meta_features_train['time'] = y_time
meta_features_train['event'] = y_event.astype(int)

# Train the meta-model
meta_model = CoxPHFitter(penalizer=0.1)
meta_model.fit(meta_features_train, duration_col='time', event_col='event')

print("\nMeta-Model Summary (shows importance of base models):")
meta_model.print_summary()

# Evaluate the stacked ensemble OOF predictions
stacked_oof_risk = meta_model.predict_partial_hazard(meta_features_train.drop(['time', 'event'], axis=1)).values.flatten()
stacked_c_index, _, _, _, _ = concordance_index_censored(y_event, y_time, stacked_oof_risk)
print(f"\nStacked Ensemble OOF C-Index: {stacked_c_index:.4f}")
if USE_WANDB: wandb.log({"stacked_oof_c_index": stacked_c_index})

# ============================================================
# 5. GENERATE FINAL SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

# Create the test set for the meta-model
meta_features_test = pd.DataFrame({
    f"{model}_risk_rank": rankdata(test_preds[model]['risk'])
    for model in CONFIG['model_list']
})

# Generate final predictions from the stacked ensemble
final_test_risk = meta_model.predict_partial_hazard(meta_features_test).values.flatten()

# --- Convert final risk to probabilities for submission ---
# This is a heuristic. A full solution would involve predicting survival functions
# for each base model and stacking those, which is more complex.
final_probs_scaled = scaler.fit_transform(final_test_risk.reshape(-1, 1)).flatten()
final_probs = 1 / (1 + np.exp(-final_probs_scaled))

submission = pd.DataFrame({'event_id': test_df['event_id']})
for h in HORIZONS:
    # Heuristic scaling for different horizons
    # A more advanced method would train a calibrator for each horizon
    scale_factor = 0.8 + 0.4 * (h / 72)
    submission[f'prob_{h}h'] = np.clip(final_probs * scale_factor, 0.001, 0.999)

# Enforce monotonicity
for i in range(len(X_test)):
    probs = [submission.loc[i, f'prob_{h}h'] for h in HORIZONS]
    for j in range(1, 4):
        if probs[j] < probs[j-1]:
            probs[j] = probs[j-1] + 1e-4 # Ensure strict increase
    for j, h in enumerate(HORIZONS):
        submission.loc[i, f'prob_{h}h'] = probs[j]

submission_path = os.path.join(DATA_DIR, "submission_v4.csv")
submission.to_csv(submission_path, index=False)

print(f"\n✅ Submission file ready: {submission_path}")
print("\nFinal prediction stats:")
print(submission.describe())

if USE_WANDB:
    artifact = wandb.Artifact('submission', type='submission')
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)
    wandb.finish()