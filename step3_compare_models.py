import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)


class CustomKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        return self

    def predict_proba(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        probabilities = np.zeros((len(X_test), 2), dtype=np.float64)

        for idx, row in enumerate(X_test):
            distances = np.sqrt(np.sum((self.X_train - row) ** 2, axis=1))
            closest_indices = np.argsort(distances)[:self.k]
            closest_labels = self.y_train[closest_indices]
            prob_failure = np.mean(closest_labels == 1)
            probabilities[idx] = [1.0 - prob_failure, prob_failure]

        return probabilities

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return (probs[:, 1] >= 0.5).astype(np.int64)


df = pd.read_csv('ai4i2020_engineered.csv')
features = [
    'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
    'Torque [Nm]', 'Tool wear [min]', 'Temp_Difference', 'Power_W', 'Strain_Metric'
]

X = df[features]
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

models = {
    'Custom KNN (From Scratch)': CustomKNN(k=5),
    'KNN (Scikit-Learn)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
trained_models = {}
roc_data = {}

for name, model in models.items():
    start_time = time.perf_counter()
    model.fit(X_train_bal, y_train_bal)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)
    elapsed = time.perf_counter() - start_time

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)
    trained_models[name] = model

    results.append({
        'Model': name,
        'Accuracy (%)': round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision (%)': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall (%)': round(recall_score(y_test, y_pred) * 100, 2),
        'F1-Score (%)': round(f1_score(y_test, y_pred) * 100, 2),
        'ROC-AUC': round(roc_auc, 3),
        'Time (s)': round(elapsed, 3)
    })

results_df = pd.DataFrame(results).sort_values(by='F1-Score (%)', ascending=False)
print("\n=== MODEL COMPARISON TABLE ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\nWinning Model (by F1-Score): {best_model_name}")

prod_candidates = results_df[results_df['Model'] != 'Custom KNN (From Scratch)']
prod_model_name = prod_candidates.iloc[0]['Model']
final_pipeline = Pipeline([
    ('scaler', scaler),
    ('model', trained_models[prod_model_name])
])
joblib.dump(final_pipeline, 'predictive_model.pkl')
print(f"Saved {prod_model_name} pipeline to predictive_model.pkl for Flask API")

sns.set_theme(style="whitegrid")

melted = results_df.melt(
    id_vars='Model',
    value_vars=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
    var_name='Metric',
    value_name='Score (%)'
)
plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x='Model', y='Score (%)', hue='Metric', palette='viridis')
plt.title('Algorithm Comparison: Custom KNN vs. Standard ML Models', fontsize=14, fontweight='bold')
plt.ylim(0, 105)
plt.xticks(rotation=20, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('graph1_model_comparison.png', dpi=300)
plt.close()

plt.figure(figsize=(9, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    if name == 'Custom KNN (From Scratch)':
        plt.plot(fpr, tpr, lw=2.5, linestyle='--', label=f'{name} (AUC = {roc_auc:.3f})')
    else:
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Guess')
plt.title('ROC Curves: Failure Detection vs. False Alarms', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate (False Alarms)')
plt.ylabel('True Positive Rate (Caught Breakdowns)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('graph2_roc_curves.png', dpi=300)
plt.close()

best_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Normal', 'Failure'],
    yticklabels=['Normal', 'Failure']
)
plt.title(f'Confusion Matrix ({best_model_name})', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('graph3_confusion_matrix.png', dpi=300)
plt.close()

tree_model = best_model if hasattr(best_model, 'feature_importances_') else trained_models['Random Forest']
imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': tree_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(imp_df['Feature'], imp_df['Importance'], color='teal')
plt.title('Mechanical Feature Importance in Failure Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('graph4_feature_importance.png', dpi=300)
plt.close()

print("All 4 result graphs saved as PNG files in your project folder.")