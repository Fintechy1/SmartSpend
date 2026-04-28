import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend

# Dark theme colors
COLORS = {
    'bg':      '#F8F9FA',
    'panel':   '#FFFFFF',
    'accent1': '#2563EB',
    'accent2': '#7C3AED',
    'accent3': '#059669',
    'accent4': '#D97706',
    'accent5': '#DC2626',
    'text':    '#111827',
    'subtext': '#6B7280',
    'grid':    '#E5E7EB'
}

def prepare_ml_data(df_full):
    """
    Prepare data for ML training.

    Args:
        df_full (pd.DataFrame): Full dataset.

    Returns:
        tuple: (X, y, feature_names, encoders_dict)
    """
    # LabelEncode categorical features
    encoders = {}
    for col in ['category', 'gender', 'state']:
        le = LabelEncoder()
        df_full[col + '_enc'] = le.fit_transform(df_full[col])
        encoders[col] = le

    # Compute global features
    amt_zscore = (df_full['amt'] - df_full['amt'].mean()) / df_full['amt'].std()
    category_count = df_full['category'].map(df_full['category'].value_counts())

    # Features
    features = ['category_enc', 'gender_enc', 'state_enc', 'hour', 'month_num',
                'year', 'quarter', 'is_weekend', 'amt', 'amt_log', 'amt_zscore', 'category_count']
    X = df_full[features]
    y = df_full['is_fraud']

    # Drop NaN
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y, features, encoders

def train_models(df_full):
    """
    Train fraud detection models.

    Args:
        df_full (pd.DataFrame): Full dataset.

    Returns:
        dict: Results dictionary.
    """
    X, y, feature_names, encoders_dict = prepare_ml_data(df_full)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model configs
    models_config = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': ['balanced']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced']
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 5],
                'subsample': [0.6, 0.8, 1.0]
            }
        },
        'LGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 10],
                'num_leaves': [15, 31, 63],
                'class_weight': ['balanced']
            }
        }
    }

    results = {}

    for name, config in models_config.items():
        print(f"Training {name}...")

        # RandomizedSearchCV
        search = RandomizedSearchCV(
            config['model'], config['params'], n_iter=20, cv=3,
            scoring='f1', random_state=42, n_jobs=-1
        )
        search.fit(X_train_scaled, y_train)

        # Best model
        best_model = search.best_estimator_

        # Predict probabilities
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        # Tune threshold
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_test, y_pred))

        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        # Final predictions with best threshold
        y_pred = (y_pred_proba >= best_thresh).astype(int)

        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

        # Print results
        print(f"  Best params: {search.best_params_}")
        print(f"  AUC: {auc:.3f}, F1: {best_f1:.3f}, Threshold: {best_thresh:.2f}")

        results[name] = {
            'model': best_model,
            'thresh': best_thresh,
            'auc': auc,
            'f1': best_f1,
            'precision': precision,
            'recall': recall,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'best_params': search.best_params_
        }

    # Save models
    os.makedirs('models', exist_ok=True)
    for name, res in results.items():
        with open(f'models/{name.lower()}.pkl', 'wb') as f:
            pickle.dump(res['model'], f)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders_dict, f)

    return results

def score_user_transactions(df_user, results, encoders_dict, scaler):
    """
    Score user transactions for fraud.

    Args:
        df_user (pd.DataFrame): User transaction data.
        results (dict): Training results.
        encoders_dict (dict): Label encoders.
        scaler (StandardScaler): Fitted scaler.

    Returns:
        pd.DataFrame: DataFrame with fraud scores.
    """
    # Encode
    df_encoded = df_user.copy()
    for col, encoder in encoders_dict.items():
        if col + '_enc' not in df_encoded.columns:
            df_encoded[col + '_enc'] = encoder.transform(df_encoded[col])

    # Compute features
    df_encoded['amt_zscore'] = (df_encoded['amt'] - df_encoded['amt'].mean()) / df_encoded['amt'].std()
    df_encoded['category_count'] = df_encoded['category'].map(df_encoded['category'].value_counts())

    features = ['category_enc', 'gender_enc', 'state_enc', 'hour', 'month_num',
                'year', 'quarter', 'is_weekend', 'amt', 'amt_log', 'amt_zscore', 'category_count']
    X = df_encoded[features]
    X_scaled = scaler.transform(X)

    # Best model by AUC
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    best_thresh = results[best_model_name]['thresh']

    # Score
    fraud_prob = best_model.predict_proba(X_scaled)[:, 1]
    fraud_predicted = (fraud_prob >= best_thresh).astype(int)

    df_user = df_user.copy()
    df_user['fraud_prob'] = fraud_prob
    df_user['fraud_predicted'] = fraud_predicted

    print(f"Flagged {fraud_predicted.sum()} transactions as potential fraud.")

    return df_user

def plot_fraud_charts(results, save_dir='charts'):
    """
    Plot fraud detection charts.

    Args:
        results (dict): Training results.
        save_dir (str): Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    for ax in axes:
        ax.set_facecolor(COLORS['panel'])

    model_names = list(results.keys())
    for i, name in enumerate(model_names):
        cm = results[name]['cm']
        ax = axes[i]
        ax.imshow(cm, cmap='Blues', alpha=0.8)
        ax.set_title(f'{name}', color=COLORS['text'], fontsize=12)

        # Annotate
        for j in range(2):
            for k in range(2):
                count = cm[j, k]
                pct = count / cm.sum() * 100
                ax.text(k, j, f'{count}\n({pct:.1f}%)',
                       ha='center', va='center', color=COLORS['text'])

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nLegitimate', 'Predicted\nFraud'], color=COLORS['subtext'])
        ax.set_yticklabels(['Actual\nLegitimate', 'Actual\nFraud'], color=COLORS['subtext'])
        ax.tick_params(colors=COLORS['subtext'])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Saved: confusion_matrices.png")

    # ROC curves
    plt.figure(figsize=(8, 7))
    plt.gca().set_facecolor(COLORS['panel'])
    plt.gcf().patch.set_facecolor(COLORS['bg'])

    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f'{name} (AUC={res["auc"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate', color=COLORS['text'])
    plt.ylabel('True Positive Rate', color=COLORS['text'])
    plt.title('ROC Curves', color=COLORS['text'])
    plt.legend(facecolor=COLORS['panel'], labelcolor=COLORS['text'])
    plt.tick_params(colors=COLORS['subtext'])
    plt.grid(True, alpha=0.3, color=COLORS['grid'])

    plt.savefig(f'{save_dir}/roc_curves.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Saved: roc_curves.png")

    # PR curves
    plt.figure(figsize=(8, 7))
    plt.gca().set_facecolor(COLORS['panel'])
    plt.gcf().patch.set_facecolor(COLORS['bg'])

    for name, res in results.items():
        thresh_idx = np.argmin(np.abs(res['precision_curve'] - res['thresh']))
        plt.plot(res['recall_curve'], res['precision_curve'], label=f'{name} (F1={res["f1"]:.3f})')
        plt.scatter(res['recall_curve'][thresh_idx], res['precision_curve'][thresh_idx],
                   marker='*', s=100, color=COLORS['accent5'], zorder=5)

    plt.xlabel('Recall', color=COLORS['text'])
    plt.ylabel('Precision', color=COLORS['text'])
    plt.title('Precision-Recall Curves', color=COLORS['text'])
    plt.legend(facecolor=COLORS['panel'], labelcolor=COLORS['text'])
    plt.tick_params(colors=COLORS['subtext'])
    plt.grid(True, alpha=0.3, color=COLORS['grid'])

    plt.savefig(f'{save_dir}/pr_curves.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Saved: pr_curves.png")

def print_summary_table(results):
    """
    Print summary table of model results.

    Args:
        results (dict): Training results.
    """
    print("\nModel Performance Summary")
    print("=" * 50)
    print(f"{'Model':<20} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Threshold':>10}")
    print("-" * 74)
    for name, res in results.items():
        print(f"{name:<20} {res['auc']:>8.3f} {res['f1']:>8.3f} {res['precision']:>10.3f} {res['recall']:>8.3f} {res['thresh']:>10.2f}")