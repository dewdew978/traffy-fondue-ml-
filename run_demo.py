"""
run_demo.py
-----------
สคริปต์จำลองรันระบบครบวงจร (End-to-End Pipeline Demo) ด้วย 11 Features หน้างาน:
1. สร้างข้อมูลจำลอง Traffy Fondue 1,000 เคส พร้อม 11 Features
2. ฝึกสอน LightGBM ทำนายเวลาซ่อม (Duration Days)
3. ป้อนผลลัพธ์เข้า Actionable Decision Layer (คำนวณ Public Impact Score)
4. สรุปจัดกลุ่มตาม Action Matrix ของ กทม.
"""

import sys
import os
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import lightgbm as lgb
from src.decision_layer import BMADecisionLayer

# กำหนด Seed ให้ผลลัพธ์คงที่
np.random.seed(42)
random.seed(42)

DISTRICTS = ['ป้อมปราบศัตรูพ่าย', 'สัมพันธวงศ์', 'ดินแดง', 'วัฒนา', 'คลองเตย', 'จตุจักร', 'บางรัก', 'หนองจอก', 'บางแค']
MAIN_TYPES = ['ถนน', 'ทางเท้า', 'ไฟฟ้า', 'ความสะอาด', 'ท่อระบายน้ำ', 'จราจร']
SUB_CATS = {
    'ถนน': ['ถนนเป็นหลุมบ่อ', 'ถนนทรุดตัว', 'ผิวถนนชำรุด'],
    'ทางเท้า': ['กระเบื้องแตก', 'ทางเท้าทรุด', 'สิ่งกีดขวางบนทางเท้า'],
    'ไฟฟ้า': ['ไฟฟ้าส่องสว่างดับ', 'สายไฟห้อยระโยงระยาง', 'เสาไฟเอียง'],
    'ความสะอาด': ['ขยะตกค้าง', 'ตัดแต่งกิ่งไม้', 'กวาดล้างทำความสะอาด'],
    'ท่อระบายน้ำ': ['ฝาท่อชำรุด', 'ท่อระบายน้ำอุดตัน', 'น้ำล้นผิวถนน'],
    'จราจร': ['สัญญาณไฟจราจรเสีย', 'ป้ายจราจรชำรุด', 'เส้นจราจรเลือนราง']
}
DEPTS = {
    'ถนน': 'ฝ่ายโยธา',
    'ทางเท้า': 'ฝ่ายโยธา',
    'ไฟฟ้า': 'การไฟฟ้านครหลวง',
    'ความสะอาด': 'ฝ่ายรักษาความสะอาดฯ',
    'ท่อระบายน้ำ': 'สำนักการระบายน้ำ',
    'จราจร': 'สำนักการจราจรและขนส่ง'
}


def generate_mock_traffy_data(n=1000):
    records = []
    base_date = pd.Timestamp('2024-01-01')

    for i in range(n):
        district = random.choice(DISTRICTS)
        main_type = random.choice(MAIN_TYPES)
        sub_cat = random.choice(SUB_CATS[main_type])
        dept = DEPTS[main_type]
        severity = random.randint(1, 5)
        comment_len = random.randint(20, 250)

        created = base_date + pd.Timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))

        # จำลองเวลาซ่อมจริง: งานขุดท่อ/ซ่อมถนนใช้เวลานานกว่าเก็บขยะ
        base_days = 2.0 if main_type in ['ทางเท้า', 'ความสะอาด'] else 5.0
        noise = np.random.exponential(scale=1.5)
        duration = round(max(0.5, base_days + noise), 2)
        finished = created + pd.Timedelta(days=duration)

        records.append({
            'ticket_id': f"TK-{10000 + i}",
            'district': district,
            'main_type': main_type,
            'sub_category': sub_cat,
            'predicted_dept': dept,
            'severity': severity,
            'comment_len': comment_len,
            'timestamp': created.strftime('%Y-%m-%d %H:%M:%S'),
            'last_activity': finished.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_days': duration,
            'state': 'เสร็จสิ้น'
        })

    return pd.DataFrame(records)


def main():
    print("=" * 65)
    print("[START] TRAFFY FONDUE ML (11 FEATURES) + DECISION LAYER DEMO")
    print("=" * 65)

    # 1. สร้างข้อมูลจำลอง
    print("\n[Step 1] เตรียมข้อมูลจำลอง 1,000 เคส พร้อม 11 Features...")
    df = generate_mock_traffy_data(1000)

    # 2. ทำ Preprocessing สกัด Temporal Features
    df['created_at'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['month'] = df['created_at'].dt.month
    df['hour'] = df['created_at'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rainy_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)

    features = [
        'district', 'main_type', 'sub_category', 'predicted_dept',
        'severity', 'comment_len', 'day_of_week', 'is_weekend',
        'month', 'is_rainy_season', 'hour'
    ]
    cat_cols = ['district', 'main_type', 'sub_category', 'predicted_dept']
    for c in cat_cols:
        df[c] = df[c].astype('category')

    # แบ่ง Train (80%) / Test (20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train = train_df[features], np.log1p(train_df['duration_days'])
    X_test = test_df[features]

    # 3. เทรน LightGBM
    print("\n[Step 2] กำลังฝึกสอนโมเดล LightGBM บน 11 Features...")
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.train(params, train_data, num_boost_round=100)

    # ทำนายเวลา (Predicted Days)
    test_df['predicted_days'] = np.expm1(model.predict(X_test)).round(2)
    print("   -> ฝึกสอนโมเดลสำเร็จและทำนายระยะเวลาซ่อมเรียบร้อย")

    # 4. ส่งผลเข้า Actionable Decision Layer
    print("\n[Step 3] ส่งเข้า Actionable Decision Layer เพื่อคำนวณ Public Impact Score...")
    density_file = os.path.join(os.path.dirname(__file__), "data", "external", "bkk_population_density.csv")
    decision_engine = BMADecisionLayer(density_file if os.path.exists(density_file) else None)

    results = []
    for _, row in test_df.iterrows():
        triage = decision_engine.evaluate_ticket(
            severity=row['severity'],
            district=row['district'],
            predicted_days=row['predicted_days']
        )
        results.append({
            'Ticket ID': row['ticket_id'],
            'เขต': row['district'],
            'ประเภทหลัก': row['main_type'],
            'งานย่อย': row['sub_category'],
            'ฝ่าย': row['predicted_dept'],
            'Severity': row['severity'],
            'เวลากายภาพทำนาย (วัน)': triage['predicted_days'],
            'Pop Density': triage['pop_density'],
            'Public Impact Score': triage['public_impact_score'],
            'กลุ่มปฏิบัติการ': triage['category'],
            'Priority': triage['priority_level']
        })

    result_df = pd.DataFrame(results).sort_values(by='Public Impact Score', ascending=False)

    print("\n[Step 4] สรุปผลการจัดสรรคิวงาน (5 อันดับแรกที่เร่งด่วนที่สุด):")
    print(result_df[['Ticket ID', 'เขต', 'ประเภทหลัก', 'งานย่อย', 'ฝ่าย', 'Severity', 'เวลากายภาพทำนาย (วัน)', 'Public Impact Score', 'กลุ่มปฏิบัติการ']].head(5).to_string(index=False))

    print("\n[Step 5] สถิติจำนวนเคสตาม Action Matrix 4 กลุ่ม:")
    print(result_df['กลุ่มปฏิบัติการ'].value_counts().to_string())

    # บันทึกไฟล์ผลลัพธ์
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_triage_results.csv")
    result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n[DONE] บันทึกผลลัพธ์การคัดกรองไว้ที่: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
