"""
PKL Match Winner Predictor - Model Training
Run this after creating the dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("=" * 60)
print("🏆 PKL MATCH WINNER PREDICTOR - MODEL TRAINING")
print("=" * 60)

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('data/pkl_ml_dataset.csv')
print(f"✅ Loaded {len(df)} matches")
print(f"✅ Columns: {list(df.columns)}")

# Prepare features
print("\n🔧 Preparing features...")

# Encode team names
le = LabelEncoder()
all_teams = pd.concat([df['team_a'], df['team_b']]).unique()
le.fit(all_teams)

df['team_a_encoded'] = le.transform(df['team_a'])
df['team_b_encoded'] = le.transform(df['team_b'])

# Select features that actually exist in the dataset
feature_columns = [
    'team_a_win_pct_prev', 'team_b_win_pct_prev',
    'team_a_points_prev', 'team_b_points_prev',
    'team_a_current_wins', 'team_b_current_wins',
    'team_a_strength', 'team_b_strength',
    'team_a_encoded', 'team_b_encoded'
]

# Verify all features exist
available_features = [col for col in feature_columns if col in df.columns]
print(f"✅ Using {len(available_features)} features: {available_features}")

X = df[available_features]
y = df['target']  # target column exists in your dataset

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target distribution:")
print(y.value_counts(normalize=True))

# Split data
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Training set: {len(X_train)} matches")
print(f"📊 Test set: {len(X_test)} matches")

# Scale features
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and encoder
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("✅ Scaler and encoder saved")

# Train multiple models
print("\n🤖 Training multiple models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False)
}

results = {}

for name, model in models.items():
    print(f"\n📈 Training {name}...")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  ✅ Test Accuracy: {accuracy:.4f}")
    print(f"  ✅ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   CV Accuracy: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']:.4f})")

# Hyperparameter tuning for best model
if best_model_name == 'Random Forest':
    print("\n🔧 Tuning Random Forest...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    print(f"✅ Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test_scaled)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    print(f"✅ Tuned accuracy: {tuned_accuracy:.4f}")

elif best_model_name == 'XGBoost':
    print("\n🔧 Tuning XGBoost...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    print(f"✅ Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test_scaled)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    print(f"✅ Tuned accuracy: {tuned_accuracy:.4f}")

# Save best model
joblib.dump(best_model, 'models/latest_model.pkl')
print(f"\n✅ Best model saved to models/latest_model.pkl")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\n🌟 Feature Importance:")
    importance = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()
    print("✅ Feature importance plot saved to models/feature_importance.png")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved to models/confusion_matrix.png")

# Save metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': results[best_model_name]['accuracy'],
    'cv_accuracy': results[best_model_name]['cv_mean'],
    'n_matches': len(df),
    'n_features': len(available_features),
    'features': str(available_features),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

pd.DataFrame([metadata]).to_csv('models/model_metadata.csv', index=False)
print("\n✅ Model metadata saved to models/model_metadata.csv")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\n📁 Files created in 'models' directory:")
print("   - scaler.pkl")
print("   - label_encoder.pkl")
print("   - latest_model.pkl")
print("   - feature_importance.png")
print("   - confusion_matrix.png")
print("   - model_metadata.csv")