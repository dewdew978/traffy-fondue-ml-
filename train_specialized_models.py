"""
train_specialized_models.py
---------------------------
ฝึกสอนระบบ Specialized Sub-models (Mixture of Experts) แยกตามหน่วยงาน กทม.:
1. ฝ่ายโยธา (งานกายภาพ ถนน ทางเท้า หลุมบ่อ)
2. ฝ่ายรักษาความสะอาดฯ (งานขยะ กิ่งไม้ กวาดล้าง)
3. ฝ่ายเทศกิจ (งานกีดขวางทางเท้า จอดรถ หาบเร่)
4. ฝ่ายสิ่งแวดล้อมฯ (งานมลพิษ กลิ่น เสียง ฝุ่น)
5. สำนักการระบายน้ำ (งานท่อระบายน้ำ น้ำท่วม)
6. General Model (โมเดลกลางสำรองสำหรับฝ่ายอื่นๆ หรือเคสทั่วไป)

พร้อมเปรียบเทียบผลลัพธ์ Head-to-Head บน Test Set (ปี 2024):
โมเดลเดี่ยว (Single Model) VS โมเดลเฉพาะหน่วยงาน (Specialized Sub-models)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SPECIALIZED_DIR = os.path.join(MODELS_DIR, "specialized")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
SINGLE_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_traffy_real.txt")
CATEGORIES_SINGLE_PATH = os.path.join(MODELS_DIR, "categories.json")

os.makedirs(SPECIALIZED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET_DEPTS = [
    'ฝ่ายโยธา',
    'ฝ่ายรักษาความสะอาดฯ',
    'ฝ่ายเทศกิจ',
    'ฝ่ายสิ่งแวดล้อมฯ',
    'สำนักการระบายน้ำ'
]

DEPT_SLUG_MAP = {
    'ฝ่ายโยธา': 'yotha',
    'ฝ่ายรักษาความสะอาดฯ': 'cleanliness',
    'ฝ่ายเทศกิจ': 'thetsakit',
    'ฝ่ายสิ่งแวดล้อมฯ': 'environment',
    'สำนักการระบายน้ำ': 'drainage',
}

FEATURE_COLS = [
    'district', 'main_type', 'sub_category',
    'severity', 'comment_len', 'day_of_week',
    'is_weekend', 'month', 'is_rainy_season', 'hour'
]
CAT_COLS = ['district', 'main_type', 'sub_category']


def load_dataset(base_name, file_list, desc="Dataset"):
    """
    โหลดข้อมูลที่ผ่านการคลีนแล้วจาก data/processed/ (Parquet / CSV)
    หากยังไม่มีไฟล์ จะ fallback ไปอ่านและคลีนจาก raw CSVs
    """
    parquet_path = os.path.join(PROCESSED_DIR, f"{base_name}.parquet")
    csv_path = os.path.join(PROCESSED_DIR, f"{base_name}.csv")

    if os.path.exists(parquet_path):
        print(f"[+] โหลด {desc} จาก Cleaned Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"--> โหลด {desc} สำเร็จ: {len(df):,} แถว")
        return df
    elif os.path.exists(csv_path):
        print(f"[+] โหลด {desc} จาก Cleaned CSV: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"--> โหลด {desc} สำเร็จ: {len(df):,} แถว")
        return df

    # Fallback to cleaning raw files
    return load_and_concat(file_list, desc)


def load_and_concat(file_list, desc="Dataset"):
    dfs = []
    print(f"\n[+] กำลังโหลดและสกัด Features สำหรับ {desc} ({len(file_list)} ไฟล์)...")
    for f in file_list:
        fname = os.path.basename(f)
        try:
            cleaned = clean_traffy_data(f, max_days_threshold=60.0)
            dfs.append(cleaned)
            print(f"    - {fname}: {len(cleaned):,} เคส")
        except Exception as e:
            print(f"    [!] ข้อผิดพลาดในการอ่าน {fname}: {e}")
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        print(f"--> รวม {desc} สำเร็จ: {len(merged):,} แถว")
        return merged
    return pd.DataFrame()


def train_single_submodel(dept_name, train_subset, val_subset):
    """ฝึกสอน LightGBM เฉพาะของหน่วยงานนั้นๆ"""
    print(f"\n--- [กำลังฝึกสอน Specialized Model: {dept_name}] ---")
    print(f"    - ข้อมูล Train: {len(train_subset):,} แถว | Validation: {len(val_subset):,} แถว")

    train_df = train_subset.copy()
    val_df = val_subset.copy()

    cat_mappings = {}
    for c in CAT_COLS:
        cats = sorted(list(train_df[c].dropna().unique()))
        train_df[c] = pd.Categorical(train_df[c], categories=cats)
        val_df[c] = pd.Categorical(val_df[c], categories=cats)
        cat_mappings[c] = [str(x) for x in cats]

    slug = DEPT_SLUG_MAP.get(dept_name, dept_name)
    cat_save_path = os.path.join(SPECIALIZED_DIR, f"categories_{slug}.json")
    with open(cat_save_path, "w", encoding="utf-8") as f:
        json.dump(cat_mappings, f, ensure_ascii=False, indent=2)

    X_train = train_df[FEATURE_COLS]
    y_train = np.log1p(train_df['duration_days'])

    X_val = val_df[FEATURE_COLS]
    y_val = np.log1p(val_df['duration_days'])

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=CAT_COLS)

    # ปรับ Hyperparameters ตามลักษณะงานของแต่ละฝ่าย
    # ฝ่ายโยธางานซับซ้อนกว่าใช้ num_leaves 63, ฝ่ายรักษาความสะอาด/เทศกิจ งานตรงไปตรงมากว่าใช้ num_leaves 31
    num_leaves = 63 if dept_name in ['ฝ่ายโยธา', 'general'] else 31

    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': num_leaves,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'cat_smooth': 10.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=800,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)]
    )

    model_path = os.path.join(SPECIALIZED_DIR, f"model_{slug}.txt")
    model.save_model(model_path)
    print(f"    --> บันทึกโมเดลสำเร็จ: {model_path}")
    return model, cat_mappings


def main():
    print("=" * 75)
    print("[START] ฝึกสอน Specialized Sub-models แยกตามหน่วยงาน กทม.")
    print("=" * 75)

    train_files = sorted(glob.glob(os.path.join(RAW_DIR, "bangkok_2023-*.csv")))
    val_files = [os.path.join(RAW_DIR, f"bangkok_2024-0{m}.csv") for m in [1, 2, 3]]
    test_files = [os.path.join(RAW_DIR, f"bangkok_2024-0{m}.csv") for m in [4, 5, 6]]

    # 1. โหลดข้อมูล (ดึงจาก Cleaned Parquet ใน data/processed/ ด้วยความเร็วสูง)
    train_df = load_dataset("train_cleaned", train_files, "Train Set (ปี 2023)")
    val_df = load_dataset("val_cleaned", val_files, "Validation Set (ต้นปี 2024)")
    test_df = load_dataset("test_cleaned", test_files, "Test Set (กลางปี 2024)")

    # 2. ฝึกสอนโมเดลเฉพาะทางทั้ง 5 ฝ่ายหลัก
    specialist_models = {}
    specialist_cats = {}

    for dept in TARGET_DEPTS:
        train_sub = train_df[train_df['predicted_dept'] == dept]
        val_sub = val_df[val_df['predicted_dept'] == dept]
        if len(train_sub) > 500:
            m, cats = train_single_submodel(dept, train_sub, val_sub)
            specialist_models[dept] = m
            specialist_cats[dept] = cats

    # 3. ฝึกสอน General Model (Fallback) บนข้อมูลทั้งหมด
    general_model, general_cats = train_single_submodel("general", train_df, val_df)
    specialist_models["general"] = general_model
    specialist_cats["general"] = general_cats

    with open(os.path.join(SPECIALIZED_DIR, "dept_map.json"), "w", encoding="utf-8") as f:
        json.dump(DEPT_SLUG_MAP, f, ensure_ascii=False, indent=2)

    # 4. ทดสอบเปรียบเทียบ Head-to-Head บน Test Set
    print("\n" + "=" * 75)
    print("[Step 3] วัดผลเปรียบเทียบ: โมเดลเดี่ยว (Single) vs โมเดลเฉพาะหน่วยงาน (Specialized)")
    print("=" * 75)

    # 4.1 ทำนายด้วย Single Model ตัวเดิม
    single_model = lgb.Booster(model_file=SINGLE_MODEL_PATH)
    with open(CATEGORIES_SINGLE_PATH, "r", encoding="utf-8") as f:
        single_cats = json.load(f)

    test_single_df = test_df.copy()
    for c in ['district', 'main_type', 'sub_category', 'predicted_dept']:
        k_cats = single_cats.get(c, [])
        test_single_df[c] = pd.Categorical(test_single_df[c], categories=k_cats)

    single_11_cols = [
        'district', 'main_type', 'sub_category', 'predicted_dept',
        'severity', 'comment_len', 'day_of_week', 'is_weekend',
        'month', 'is_rainy_season', 'hour'
    ]
    single_preds_log = single_model.predict(test_single_df[single_11_cols])
    test_df['single_pred_days'] = np.expm1(single_preds_log)

    # 4.2 ทำนายด้วย Specialized Sub-models (Routing ตาม predicted_dept)
    test_df['specialized_pred_days'] = 0.0
    test_df['model_routed'] = ''

    print("[*] กำลัง Route ข้อมูล Test Set เข้าสู่โมเดลเฉพาะทางแบบ Vectorized...")
    for dept_val, group in test_df.groupby('predicted_dept'):
        dept_str = str(dept_val)
        target_key = dept_str if dept_str in specialist_models else "general"
        model = specialist_models[target_key]
        cats = specialist_cats[target_key]

        sub_df = group[FEATURE_COLS].copy()
        for c in CAT_COLS:
            k_cats = cats.get(c, [])
            sub_df[c] = pd.Categorical(sub_df[c], categories=k_cats)

        preds_log = model.predict(sub_df[FEATURE_COLS])
        preds_days = np.clip(np.expm1(preds_log), 0.1, None)

        test_df.loc[group.index, 'specialized_pred_days'] = preds_days
        test_df.loc[group.index, 'model_routed'] = target_key

    # 5. สรุปผลเปรียบเทียบภาพรวม (Overall Comparison)
    actual_days = test_df['duration_days'].values

    single_mae = mean_absolute_error(actual_days, test_df['single_pred_days'])
    single_med_ae = median_absolute_error(actual_days, test_df['single_pred_days'])

    spec_mae = mean_absolute_error(actual_days, test_df['specialized_pred_days'])
    spec_med_ae = median_absolute_error(actual_days, test_df['specialized_pred_days'])

    diff_single = np.abs(actual_days - test_df['single_pred_days'])
    diff_spec = np.abs(actual_days - test_df['specialized_pred_days'])

    single_acc3 = np.mean(diff_single <= 3.0) * 100
    spec_acc3 = np.mean(diff_spec <= 3.0) * 100

    print("\n" + "=" * 65)
    print("🏆 สรุปผลการแข่งขันภาพรวม (Overall Performance)")
    print("=" * 65)
    print(f"ตัวชี้วัด                        โมเดลเดี่ยว (Single)   โมเดลเฉพาะทาง (Specialized)")
    print(f"1. MAE (คลาดเคลื่อนเฉลี่ย)      : {single_mae:7.2f} วัน        {spec_mae:7.2f} วัน")
    print(f"2. Median AE (มัธยฐานคลาดเคลื่อน): {single_med_ae:7.2f} วัน        {spec_med_ae:7.2f} วัน")
    print(f"3. ความแม่นยำในกรอบ ±3 วัน      : {single_acc3:7.2f}%         {spec_acc3:7.2f}%")
    print("=" * 65)

    # 6. เจาะลึกรายหน่วยงาน (Per-Department Breakdown)
    dept_stats = []
    all_depts = list(test_df['predicted_dept'].unique())

    for d in TARGET_DEPTS + [dept for dept in all_depts if dept not in TARGET_DEPTS]:
        sub = test_df[test_df['predicted_dept'] == d]
        if len(sub) == 0:
            continue
        sub_act = sub['duration_days'].values
        sub_s_mae = mean_absolute_error(sub_act, sub['single_pred_days'])
        sub_s_med = median_absolute_error(sub_act, sub['single_pred_days'])
        sub_p_mae = mean_absolute_error(sub_act, sub['specialized_pred_days'])
        sub_p_med = median_absolute_error(sub_act, sub['specialized_pred_days'])
        improvement = ((sub_s_mae - sub_p_mae) / sub_s_mae) * 100

        dept_stats.append({
            'หน่วยงาน': d,
            'จำนวนเคส': len(sub),
            'Single MAE': round(sub_s_mae, 2),
            'Specialized MAE': round(sub_p_mae, 2),
            'Single MedAE': round(sub_s_med, 2),
            'Spec MedAE': round(sub_p_med, 2),
            'MAE Improvement (%)': round(improvement, 2)
        })

    dept_df = pd.DataFrame(dept_stats)
    print("\n[เจาะลึกเปรียบเทียบผลลัพธ์รายหน่วยงาน (Per-Department Breakdown)]:")
    print(dept_df.to_string(index=False))

    # บันทึกรายงานเปรียบเทียบ
    report_path = os.path.join(REPORTS_DIR, "specialized_vs_single_comparison.csv")
    dept_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n[DONE] บันทึกรายงานเปรียบเทียบไว้ที่: {report_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
