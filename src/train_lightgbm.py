"""
train_lightgbm.py
-----------------
สคริปต์ฝึกสอนโมเดล LightGBM สำหรับทำนายระยะเวลาซ่อม (Duration Days)
- ใช้ Log-Transformation บน Target
- ใช้ Objective 'regression_l1' (MAE Loss) ป้องกัน Outlier
- รองรับ Categorical Feature แบบ Native
- บันทึกโมเดลไว้ในโฟลเดอร์ models/
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, median_absolute_error


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, model_output_path: str = "models/lightgbm_model.txt"):
    feature_cols = [
        'district', 'type', 'day_of_week', 'month', 'hour',
        'is_weekend', 'is_rainy_season'
    ]
    
    # เก็บเฉพาะคอลัมน์ที่มีอยู่จริงใน DataFrame
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    cat_cols = [c for c in ['district', 'type'] if c in feature_cols]

    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        val_df[col] = val_df[col].astype('category')

    # Target: Log-transform duration_days
    X_train, y_train = train_df[feature_cols], np.log1p(train_df['duration_days'])
    X_val, y_val = val_df[feature_cols], np.log1p(val_df['duration_days'])

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)

    params = {
        'objective': 'regression_l1',   # MAE loss
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'cat_smooth': 10.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    print("Starting LightGBM training...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)]
    )

    # ประเมินผล (แปลงกลับด้วย expm1)
    preds_log = model.predict(X_val)
    preds_days = np.expm1(preds_log)
    actual_days = val_df['duration_days'].values

    mae = mean_absolute_error(actual_days, preds_days)
    med_ae = median_absolute_error(actual_days, preds_days)

    print("\n--- Model Evaluation Results ---")
    print(f"Validation MAE: {mae:.2f} วัน")
    print(f"Validation Median AE: {med_ae:.2f} วัน")

    # บันทึกโมเดล
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save_model(model_output_path)
    print(f"Model saved successfully at: {model_output_path}")

    return model


if __name__ == "__main__":
    print("Training script ready. Run train_model() with your prepared datasets.")
