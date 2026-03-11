"""
SHAP and LIME explainability for the KNN energy consumption forecasting model.
Generates:
  reports/shap_summary.png      — SHAP beeswarm summary across all test predictions
  reports/shap_waterfall.png    — SHAP waterfall plot for a single test prediction
  reports/lime_explanation.png  — LIME bar chart for a single test prediction
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# allow running from project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocess import load_data, add_features, split_data, fetch_temperature

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')


def build_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_scaled, y_train)
    return knn, scaler


def make_predict_fn(knn, scaler):
    """Wrap knn so KernelExplainer can call it on raw (unscaled) input."""
    def predict(X):
        return knn.predict(scaler.transform(X))
    return predict


def run_shap(X_train, X_test, features, predict_fn, n_background=100, n_explain=50):
    """
    Use KernelExplainer (model-agnostic).
    Background: a random sample from training set.
    Explain: first n_explain rows of the test set.
    """
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    background = X_train[bg_idx]

    explainer = shap.KernelExplainer(predict_fn, background, feature_names=features)
    X_explain = X_test[:n_explain]
    shap_values = explainer(X_explain)
    return shap_values, X_explain


def save_summary_plot(shap_values, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot → {output_path}")


def save_waterfall_plot(shap_values, idx, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved waterfall plot → {output_path}")


def run_lime(X_train, X_test, features, predict_fn, idx=0):
    """
    Use LimeTabularExplainer (model-agnostic).
    Fits a local linear model around a single prediction.
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=features,
        mode='regression',
        random_state=42,
    )
    explanation = explainer.explain_instance(
        data_row=X_test[idx],
        predict_fn=predict_fn,
        num_features=len(features),
    )
    return explanation


def save_lime_plot(explanation, output_path):
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(9, 5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved LIME explanation → {output_path}")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    temperature = fetch_temperature('2006-01-01', '2017-12-31')
    data = add_features(df, temperature)
    train_data, test_data, features, target = split_data(data)

    X_train = train_data[features].values
    y_train = train_data[target].values
    X_test = test_data[features].values

    print("Training KNN model...")
    knn, scaler = build_model(X_train, y_train)
    predict_fn = make_predict_fn(knn, scaler)

    print("Computing SHAP values (KernelExplainer — this may take a minute)...")
    shap_values, X_explain = run_shap(X_train, X_test, features, predict_fn)

    summary_path = os.path.join(REPORTS_DIR, 'shap_summary.png')
    waterfall_path = os.path.join(REPORTS_DIR, 'shap_waterfall.png')

    save_summary_plot(shap_values, summary_path)
    save_waterfall_plot(shap_values, idx=0, output_path=waterfall_path)

    print("Computing LIME explanation...")
    lime_exp = run_lime(X_train, X_test, features, predict_fn, idx=0)
    save_lime_plot(lime_exp, os.path.join(REPORTS_DIR, 'lime_explanation.png'))

    print("\nDone. Reports saved to reports/")


if __name__ == '__main__':
    main()
