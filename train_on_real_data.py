"""
train_on_real_data.py
---------------------
สคริปต์ฝึกสอนโมเดล LightGBM ด้วย 11 Features หน้างาน จากข้อมูลจริง Traffy Fondue กทม.:
- Features ทั้ง 11 ตัว:
  1. district (เขต 50 เขต)
  2. main_type (ประเภทปัญหาหลัก)
  3. sub_category (เนื้องานย่อย)
  4. predicted_dept (ฝ่ายที่รับผิดชอบ)
  5. severity (ระดับความรุนแรง 1-5)
  6. comment_len (ความยาวข้อความ)
  7. day_of_week (วันในสัปดาห์ 0-6)
  8. is_weekend (เสาร์-อาทิตย์)
  9. month (เดือนที่แจ้ง 1-12)
  10. is_rainy_season (หน้าฝน พ.ค.-ต.ค.)
  11. hour (เวลาที่แจ้ง 0-23 น.)

- Target: duration_days (ระยะเวลาซ่อมเสร็จจริง)
- Train Set: ปี 2023 (12 เดือน)
- Validation Set: 2024-01 ถึง 2024-03
- Test Set: 2024-04 ถึง 2024-06
"""

import sys
import os
import glob
import time
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from src.data_preprocessing import clean_traffy_data
from src.decision_layer import BMADecisionLayer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
DENSITY_FILE = os.path.join(BASE_DIR, "data", "external", "bkk_population_density.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_and_concat(file_list, desc="Dataset"):
    dfs = []
    print(f"\n[+] กำลังโหลดและสกัด 11 Features สำหรับ {desc} ({len(file_list)} ไฟล์)...")
    for f in file_list:
        fname = os.path.basename(f)
        try:
            cleaned = clean_traffy_data(f, max_days_threshold=60.0)
            dfs.append(cleaned)
            print(f"    - {fname}: {len(cleaned):,} เคสที่สกัด Features สมบูรณ์")
        except Exception as e:
            print(f"    [!] ข้อผิดพลาดในการอ่าน {fname}: {e}")
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        print(f"--> รวม {desc} สำเร็จ: {len(merged):,} แถว")
        return merged
    return pd.DataFrame()


def main():
    print("=" * 70)
    print("[START] ฝึกสอน LightGBM ด้วย 11 Features หน้างาน (Traffy Fondue)")
    print("=" * 70)

    train_files = sorted(glob.glob(os.path.join(RAW_DIR, "bangkok_2023-*.csv")))
    val_files = [os.path.join(RAW_DIR, f"bangkok_2024-0{m}.csv") for m in [1, 2, 3]]
    test_files = [os.path.join(RAW_DIR, f"bangkok_2024-0{m}.csv") for m in [4, 5, 6]]

    # 1. โหลดและทำความสะอาดข้อมูลทั้ง 3 ชุด
    train_df = load_and_concat(train_files, "Train Set (ปี 2023)")
    val_df = load_and_concat(val_files, "Validation Set (ต้นปี 2024)")
    test_df = load_and_concat(test_files, "Test Set (กลางปี 2024)")

    # 2. จัดเตรียม 11 Features
    feature_cols = [
        'district', 'main_type', 'sub_category', 'predicted_dept',
        'severity', 'comment_len', 'day_of_week', 'is_weekend',
        'month', 'is_rainy_season', 'hour'
    ]
    cat_cols = ['district', 'main_type', 'sub_category', 'predicted_dept']

    print("\n[Step 2] จัดเตรียม Categorical Encodings และบันทึกลง categories.json...")
    cat_mappings = {}
    for c in cat_cols:
        # รวบรวม Categories ทั้งหมดจาก Train
        train_cats = sorted(list(train_df[c].dropna().unique()))
        train_df[c] = pd.Categorical(train_df[c], categories=train_cats)
        val_df[c] = pd.Categorical(val_df[c], categories=train_cats)
        test_df[c] = pd.Categorical(test_df[c], categories=train_cats)
        cat_mappings[c] = [str(item) for item in train_cats]

    cat_save_path = os.path.join(MODELS_DIR, "categories.json")
    with open(cat_save_path, "w", encoding="utf-8") as f:
        json.dump(cat_mappings, f, ensure_ascii=False, indent=2)
    print(f"--> บันทึกหมวดหมู่ Categorical ไว้ที่: {cat_save_path}")

    # เตรียมตัวแปร X และ y (Log-transform duration_days)
    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df['duration_days'])

    X_val = val_df[feature_cols]
    y_val = np.log1p(val_df['duration_days'])

    X_test = test_df[feature_cols]
    y_test = np.log1p(test_df['duration_days'])

    # 3. กำหนด LightGBM Dataset & Params
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)

    params = {
        'objective': 'regression_l1',   # MAE loss ทนต่อ Outlier
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'cat_smooth': 15.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    print("\n[Step 3] เริ่มต้นการฝึกสอนโมเดล LightGBM บน 11 Features...")
    start_t = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1200,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(100)]
    )
    print(f"--> เทรนเสร็จสิ้นในเวลา {time.time() - start_t:.1f} วินาที")

    # บันทึกโมเดล
    model_path = os.path.join(MODELS_DIR, "lightgbm_traffy_real.txt")
    model.save_model(model_path)
    print(f"--> บันทึกโมเดลไว้ที่: {model_path}")

    # 4. ประเมินผลบน Test Set
    print("\n[Step 4] ประเมินผลบน Test Set (กลางปี 2024)...")
    test_preds_log = model.predict(X_test)
    test_preds_days = np.expm1(test_preds_log)
    actual_days = test_df['duration_days'].values

    mae = mean_absolute_error(actual_days, test_preds_days)
    med_ae = median_absolute_error(actual_days, test_preds_days)
    r2 = r2_score(actual_days, test_preds_days)

    diff = np.abs(actual_days - test_preds_days)
    acc_1d = np.mean(diff <= 1.0) * 100
    acc_2d = np.mean(diff <= 2.0) * 100
    acc_3d = np.mean(diff <= 3.0) * 100

    print("\n" + "=" * 55)
    print("📊 ผลการประเมินความแม่นยำของโมเดล (11 Features Test Set)")
    print("=" * 55)
    print(f"1. MAE (คลาดเคลื่อนเฉลี่ย)           : {mae:.2f} วัน")
    print(f"2. Median AE (มัธยฐานความคลาดเคลื่อน): {med_ae:.2f} วัน")
    print(f"3. R2 Score (ความสัมพันธ์)           : {r2:.3f}")
    print(f"4. ความแม่นยำในกรอบ ±1 วัน (24 ชม.)  : {acc_1d:.2f}%")
    print(f"5. ความแม่นยำในกรอบ ±2 วัน (48 ชม.)  : {acc_2d:.2f}%")
    print(f"6. ความแม่นยำในกรอบ ±3 วัน           : {acc_3d:.2f}%")
    print("=" * 55)

    # 5. วิเคราะห์ Feature Importance (Gain & Split)
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')
    gain_pct = (importance_gain / np.sum(importance_gain)) * 100

    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance (Gain)': np.round(importance_gain, 1),
        'Gain (%)': np.round(gain_pct, 2),
        'Splits Count': importance_split
    }).sort_values(by='Gain (%)', ascending=False)

    print("\n[Feature Importance Ranking จาก 11 Features]:")
    print(fi_df.to_string(index=False))

    # 6. เชื่อมต่อไปยัง Actionable Decision Layer
    print("\n[Step 5] ส่งผลทำนายเข้า Actionable Decision Layer...")
    decision_engine = BMADecisionLayer(DENSITY_FILE if os.path.exists(DENSITY_FILE) else None)
    
    test_df['predicted_days'] = test_preds_days.round(2)

    # ประเมินตัวอย่าง 5,000 เคส
    sample_eval = test_df.head(5000).copy()
    triage_results = []
    for _, row in sample_eval.iterrows():
        res = decision_engine.evaluate_ticket(
            severity=int(row['severity']),
            district=row['district'],
            predicted_days=row['predicted_days']
        )
        triage_results.append({
            'ticket_id': row['ticket_id'],
            'district': row['district'],
            'main_type': row['main_type'],
            'sub_category': row['sub_category'],
            'predicted_dept': row['predicted_dept'],
            'severity': int(row['severity']),
            'predicted_days': row['predicted_days'],
            'actual_days': round(row['duration_days'], 2),
            'pop_density': res['pop_density'],
            'public_impact_score': res['public_impact_score'],
            'category': res['category'],
            'priority': res['priority_level'],
            'action': res['action_recommendation']
        })

    triage_df = pd.DataFrame(triage_results).sort_values(by='public_impact_score', ascending=False)
    out_triage_path = os.path.join(REPORTS_DIR, "real_triage_evaluation.csv")
    triage_df.to_csv(out_triage_path, index=False, encoding='utf-8-sig')

    print(f"\n[DONE] บันทึกผลลัพธ์ Triage ไว้ที่: {out_triage_path}")
    print("\n[ตัวอย่าง 5 เคสที่เร่งด่วนที่สุด]:")
    print(triage_df[['ticket_id', 'district', 'main_type', 'sub_category', 'predicted_dept', 'predicted_days', 'public_impact_score', 'category']].head(5).to_string(index=False))

    print("\n[สถิติจำนวนเคสตามกลุ่มปฏิบัติการ กทม.]:")
    print(triage_df['category'].value_counts().to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
