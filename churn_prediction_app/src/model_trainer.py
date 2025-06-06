import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras import backend as K
import tensorflow as tf

def focal_loss(gamma=2., alpha=.25):
    """Focal loss for binary classification."""
    def focal_loss_fixed(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, X_sm, y_sm):
    """
    Trains and evaluates multiple churn prediction models.
    Args:
        X_train_scaled (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target.
        X_test_scaled (pd.DataFrame): Scaled testing features.
        y_test (pd.Series): Testing target.
        X_sm (pd.DataFrame): SMOTE-resampled training features.
        y_sm (pd.Series): SMOTE-resampled training target.
    Returns:
        dict: A dictionary containing trained models.
    """
    trained_models = {}

    # ----- MODEL 1: XGBoost + SMOTE (Original X and y for SMOTE here) -----
    # Note: X_sm and y_sm passed here are already scaled from preprocessor
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_sm, y_sm)
    y_pred_xgb = xgb.predict(X_test_scaled) # Predict on scaled X_test
    print("\n🎯 XGBoost + SMOTE")
    print(classification_report(y_test, y_pred_xgb))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_xgb))
    trained_models['xgb_smote'] = xgb

    # ----- MODEL 2: ANN + Class Weights -----
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    ann = Sequential()
    ann.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train_scaled, y_train, epochs=50, batch_size=32, class_weight=class_weights, verbose=0)
    y_pred_ann = (ann.predict(X_test_scaled) > 0.5).astype("int32")
    print("\n🤖 ANN + Class Weights")
    print(classification_report(y_test, y_pred_ann))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_ann))
    trained_models['ann_class_weights'] = ann

    # ----- MODEL 3: ANN + SMOTE -----
    ann_sm = Sequential()
    ann_sm.add(Dense(64, activation='relu', input_dim=X_sm.shape[1])) # Input dim from SMOTE data
    ann_sm.add(Dense(32, activation='relu'))
    ann_sm.add(Dense(1, activation='sigmoid'))
    ann_sm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann_sm.fit(X_sm, y_sm, epochs=50, batch_size=32, verbose=0)
    y_pred_ann_sm = (ann_sm.predict(X_test_scaled) > 0.5).astype("int32")
    print("\n🔁 ANN + SMOTE")
    print(classification_report(y_test, y_pred_ann_sm))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_ann_sm))
    trained_models['ann_smote'] = ann_sm

    # ----- MODEL 4: ANN + Focal Loss -----
    ann_focal = Sequential()
    ann_focal.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
    ann_focal.add(Dense(32, activation='relu'))
    ann_focal.add(Dense(1, activation='sigmoid'))
    ann_focal.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
    ann_focal.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred_focal = (ann_focal.predict(X_test_scaled) > 0.5).astype("int32")
    print("\n🔥 ANN + Focal Loss")
    print(classification_report(y_test, y_pred_focal))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_focal))
    trained_models['ann_focal_loss'] = ann_focal

    return trained_models

def save_models(models, models_dir='models'):
    """Saves trained models to the specified directory."""
    os.makedirs(models_dir, exist_ok=True)
    for name, model in models.items():
        if 'xgb' in name:
            joblib.dump(model, os.path.join(models_dir, f'{name}_model.pkl'))
        elif 'ann' in name:
            model.save(os.path.join(models_dir, f'{name}_model.keras'))
        print(f"Model '{name}' saved.")

if __name__ == '__main__':
    # Example usage:
    # This part would typically be run once to train and save your models.
    from data_loader import load_data
    from preprocessor import preprocess_data, save_scaler
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'customer_churn.xlsx')
    models_dir = os.path.join(project_root, 'models')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    df = load_data(data_path)
    if df is not None:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Preprocess training and testing data
        X_train_scaled, y_train_processed, scaler, X_sm, y_sm, X_train_cols = preprocess_data(
            pd.concat([X_train, y_train], axis=1), is_training=True
        )
        # For X_test_scaled, we need to apply the same preprocessing steps as X_train_scaled
        # but using the *fitted* scaler from training.
        X_test_scaled, _, _, _, _, _ = preprocess_data(
            pd.concat([X_test, y_test], axis=1), is_training=False, scaler=scaler, X_train_columns=X_train_cols
        )

        # Save the scaler
        save_scaler(scaler, scaler_path)

        # Train and evaluate
        trained_models = train_and_evaluate_models(
            X_train_scaled, y_train, X_test_scaled, y_test, X_sm, y_sm
        )

        # Save all trained models
        save_models(trained_models, models_dir)