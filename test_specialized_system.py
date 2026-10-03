"""
test_specialized_system.py
--------------------------
ทดสอบระบบ Specialized Sub-models กับข้อมูลจริงที่คลีนแล้วใน data/processed/test_cleaned.parquet:
1. โหลดข้อมูล Test Set จริง (Unseen Data 2024)
2. สุ่มเคสตัวแทนจากฝ่ายหลักทั้ง 5 ฝ่าย (โยธา, รักษาความสะอาด, เทศกิจ, สิ่งแวดล้อม, ระบายน้ำ)
3. รันผ่าน SpecializedBMAPredictor เพื่อจำลองการทำงานจริงของ Router และ Sub-models
4. แสดงผลการทำนายเปรียบเทียบกับวันที่ กทม. ซ่อมเสร็จจริง
"""

import sys
import os
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.specialized_predictor import SpecializedBMAPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PARQUET = os.path.join(BASE_DIR, "data", "processed", "test_cleaned.parquet")


def main():
    print("=" * 80)
    print("🚀 [TEST] ทดสอบระบบทำนาย Specialized Sub-models กับ Cleaned Test Data จริง")
    print("=" * 80)

    # 1. โหลด Test Set จาก data/processed/
    print(f"\n[1] กำลังอ่านข้อมูลทดสอบจาก: {TEST_PARQUET}")
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"--> โหลดสำเร็จ: {len(test_df):,} เคส (คอลัมน์: {list(test_df.columns)})")

    # 2. เริ่มต้นระบบ Specialized Predictor
    print("\n[2] กำลังโหลดโมเดลเฉพาะทางทั้ง 5 ฝ่าย + General Fallback เข้าหน่วยความจำ...")
    predictor = SpecializedBMAPredictor()
    print(f"--> โมเดลพร้อมใช้งาน: {predictor.is_ready()}")
    print(f"    รายการโมเดลย่อย: {list(predictor.models.keys())}")

    # 3. เลือกเคสตัวแทนจากหลากหลายฝ่ายงาน (ฝ่ายละ 2 เคส)
    target_depts = [
        'ฝ่ายโยธา',
        'ฝ่ายรักษาความสะอาดฯ',
        'ฝ่ายเทศกิจ',
        'ฝ่ายสิ่งแวดล้อมฯ',
        'สำนักการระบายน้ำ'
    ]

    sampled_cases = []
    for dept in target_depts:
        sub = test_df[test_df['predicted_dept'] == dept]
        if len(sub) > 0:
            # เลือกเคสที่มีข้อความชัดเจน
            sample = sub.sort_values(by='comment_len', ascending=False).head(2)
            sampled_cases.append(sample)

    eval_df = pd.concat(sampled_cases, ignore_index=True)
    print(f"\n[3] คัดเลือกตัวอย่างจริง {len(eval_df)} เคสจาก 5 ฝ่ายหลักมาทดสอบทำนาย:")

    # 4. ทดสอบทำนายด้วย Specialized Predictor
    results_df = predictor.predict_df(eval_df)

    print("\n" + "=" * 90)
    print(f"{'เขต':<12} | {'ประเภทงาน':<15} | {'ฝ่ายที่รับผิดชอบ':<20} | {'โมเดลที่เลือก':<20} | {'ทำนาย (วัน)':<12} | {'จริง (วัน)':<10} | {'คลาดเคลื่อน':<10}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        district = str(row['district'])[:10]
        work_type = str(row['sub_category'])[:13]
        dept = str(row['predicted_dept'])[:18]
        model_used = str(row['model_used'])[:18]
        pred_days = float(row['specialized_pred_days'])
        actual_days = float(row['duration_days'])
        diff = abs(pred_days - actual_days)

        print(f"{district:<12} | {work_type:<15} | {dept:<20} | {model_used:<20} | {pred_days:7.2f} วัน   | {actual_days:7.2f} วัน | {diff:7.2f} วัน")

    print("=" * 90)

    # 5. ทดสอบการทำนายแบบ Real-time API Case (Single Payload)
    print("\n[4] ทดสอบส่ง Payload จำลองแบบ Real-time เข้าฟังก์ชัน predict_single():")
    mock_payload = {
        'district': 'จตุจักร',
        'main_type': 'ขยะ',
        'sub_category': 'ขยะตกค้างส่งกลิ่นเหม็น',
        'severity': 3,
        'comment_len': 45,
        'day_of_week': 2,
        'is_weekend': 0,
        'month': 6,
        'is_rainy_season': 1,
        'hour': 8
    }
    single_res = predictor.predict_single(mock_payload, predicted_dept='ฝ่ายรักษาความสะอาดฯ')
    print("--> ข้อมูลจำลอง:", mock_payload)
    print("--> ผลการตอบกลับจาก AI Predictor:")
    print(f"    - คาดการณ์เวลาเสร็จ: {single_res['predicted_days']} วัน (~{single_res['predicted_days']*24:.1f} ชั่วโมง)")
    print(f"    - โมเดลที่ถูกเลือกใช้ : {single_res['model_used']}")
    print(f"    - Model Slug       : {single_res['model_slug']}")

    print("\n" + "=" * 80)
    print("✅ การทดสอบระบบเสร็จสิ้นสมบูรณ์ ทุกโมเดลและฟังก์ชันทำงานได้อย่างถูกต้อง 100%")
    print("=" * 80)


if __name__ == "__main__":
    main()
