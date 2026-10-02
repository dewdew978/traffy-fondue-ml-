"""
test_real_pipeline.py
---------------------
ทดสอบรันระบบครบวงจรกับ "ข้อมูลจริงของประชาชน กทม." (Traffy Fondue):
1. คัดเลือกเคสข้อร้องเรียนจริง 5 เคส จากไฟล์ bangkok_2024-06.csv
2. ส่งข้อความจริงให้ Local LLM Ollama (qwen3:4b) วิเคราะห์:
   - severity (1-5)
   - sub_category (เนื้องานย่อย)
   - predicted_dept (ฝ่ายที่ควรรับผิดชอบ)
   - reason (เหตุผลประกอบ)
3. รวบรวม 11 Features หน้างาน แล้วส่งเข้าโมเดล LightGBM ทำนายจำนวนวันซ่อม
4. ป้อนเข้า Actionable Decision Layer คำนวณ Public Impact Score และจัดกลุ่ม Triage กทม.
5. เปรียบเทียบกับ "จำนวนวันที่ กทม. ซ่อมเสร็จจริง"
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.llm_severity import LLMComplaintAnalyzer
from src.decision_layer import BMADecisionLayer
from src.data_preprocessing import parse_type_hierarchy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "bangkok_2024-06.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_traffy_real.txt")
CATEGORIES_PATH = os.path.join(BASE_DIR, "models", "categories.json")
DENSITY_FILE = os.path.join(BASE_DIR, "data", "external", "bkk_population_density.csv")


def main():
    print("=" * 75)
    print("[START] ทดสอบระบบ AI + LightGBM (11 Features) กับข้อมูลจริงประชาชน กทม.")
    print("=" * 75)

    # 1. โหลดข้อมูลจริงและสุ่มเลือก 5 เคสที่มีข้อความชัดเจน
    print("\n[Step 1] คัดเลือกเคสตัวแทน 5 ปัญหาจากไฟล์ bangkok_2024-06.csv ...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    valid = df[
        (df['state'].astype(str).str.contains('เสร็จ|finish', case=False, na=False)) &
        (df['comment'].fillna('').str.len() >= 30) &
        (df['district'].notna()) &
        (df['duration_minutes_total'].notna())
    ].copy()

    # เลือกตัวอย่างปัญหาหลากหลายหมวด
    sample_tickets = valid.drop_duplicates(subset=['type']).head(5).copy()

    sample_tickets['actual_days'] = pd.to_numeric(sample_tickets['duration_minutes_total'], errors='coerce') / 1440.0
    sample_tickets['district'] = sample_tickets['district'].astype(str).str.replace('เขต', '').str.strip()

    # แยกประเภทหลัก
    parsed = sample_tickets['type'].fillna('ทั่วไป').astype(str).apply(parse_type_hierarchy)
    sample_tickets['main_type'] = [p[0] for p in parsed]
    sample_tickets['tag_sub_category'] = [p[1] for p in parsed]

    sample_tickets['created_at'] = pd.to_datetime(sample_tickets['timestamp'], errors='coerce')
    sample_tickets['day_of_week'] = sample_tickets['created_at'].dt.dayofweek
    sample_tickets['month'] = sample_tickets['created_at'].dt.month
    sample_tickets['hour'] = sample_tickets['created_at'].dt.hour
    sample_tickets['is_weekend'] = sample_tickets['day_of_week'].isin([5, 6]).astype(int)
    sample_tickets['is_rainy_season'] = sample_tickets['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    sample_tickets['comment_len'] = sample_tickets['comment'].str.len()

    # 2. โหลดโมเดล LightGBM และหมวดหมู่ Categorical
    print("\n[Step 2] โหลดโมเดล LightGBM และไฟล์ categories.json ...")
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        cat_mappings = json.load(f)

    # 3. เตรียม LLM Complaint Analyzer (Ollama) & Decision Layer
    print("\n[Step 3] เชื่อมต่อ Ollama (qwen3:4b) และ BMA Decision Layer ...")
    llm_analyzer = LLMComplaintAnalyzer(ollama_model="qwen3:4b")
    decision_engine = BMADecisionLayer(DENSITY_FILE)

    # 4. ประมวลผลทีละเคสด้วย 11 Features
    print("\n" + "=" * 75)
    print("[RESULTS] เริ่มประมวลผลทั้ง 5 เคสจริง:")
    print("=" * 75)

    summary_rows = []
    feature_cols = [
        'district', 'main_type', 'sub_category', 'predicted_dept',
        'severity', 'comment_len', 'day_of_week', 'is_weekend',
        'month', 'is_rainy_season', 'hour'
    ]

    for i, (_, row) in enumerate(sample_tickets.iterrows(), 1):
        ticket_id = row['ticket_id']
        comment = str(row['comment']).strip()
        district = row['district']
        main_type = row['main_type']
        actual_d = round(row['actual_days'], 2)

        print(f"\n--- [เคสที่ {i}/5] Ticket ID: {ticket_id} ---")
        print(f"[*] เขต: {district} | ประเภทหลัก: {main_type}")
        print(f"[*] ข้อความร้องเรียนจริง:\n   \"{comment}\"")

        # 4.1 ให้ LLM อ่านข้อความและสกัด Features
        llm_res = llm_analyzer.evaluate_comment(comment, use_ollama=True)
        severity = llm_res.get('severity', 3)
        sub_category = llm_res.get('sub_category', row['tag_sub_category'])
        predicted_dept = llm_res.get('predicted_dept', 'สำนักงานเขตทั่วไป')
        reason = llm_res.get('reason', '-')
        engine = llm_res.get('engine', '-')

        print(f"[*] [LLM ({engine})]:")
        print(f"    - Severity       : {severity}/5")
        print(f"    - Sub-category   : {sub_category}")
        print(f"    - Predicted Dept : {predicted_dept}")
        print(f"    - Rationale      : {reason}")

        # 4.2 ประกอบ 11 Features เพื่อป้อนให้ LightGBM
        ticket_features = pd.DataFrame([{
            'district': district,
            'main_type': main_type,
            'sub_category': sub_category,
            'predicted_dept': predicted_dept,
            'severity': severity,
            'comment_len': row['comment_len'],
            'day_of_week': row['day_of_week'],
            'is_weekend': row['is_weekend'],
            'month': row['month'],
            'is_rainy_season': row['is_rainy_season'],
            'hour': row['hour']
        }])

        # กำหนด Categorical Dtype ให้ตรงกับโมเดล
        for c in ['district', 'main_type', 'sub_category', 'predicted_dept']:
            known_cats = cat_mappings.get(c, [])
            val = ticket_features[c].iloc[0]
            # หากเป็นคำใหม่ที่ไม่เคยเห็น ให้ fallback เป็น 'ทั่วไป' หรือคำแรก
            if val not in known_cats:
                fallback_val = 'ทั่วไป' if 'ทั่วไป' in known_cats else (known_cats[0] if known_cats else val)
                ticket_features[c] = fallback_val
            ticket_features[c] = pd.Categorical(ticket_features[c], categories=known_cats)

        # 4.3 ทำนายเวลาซ่อมด้วย LightGBM
        pred_log = model.predict(ticket_features[feature_cols])
        pred_d = float(np.expm1(pred_log)[0].round(2))

        # 4.4 คำนวณ Public Impact Score และ Triage
        triage = decision_engine.evaluate_ticket(
            severity=severity,
            district=district,
            predicted_days=pred_d
        )

        impact_score = triage['public_impact_score']
        category = triage['category']
        action = triage['action_recommendation']

        print(f"[*] [LightGBM ML]:")
        print(f"    - เวลาทำนาย (ML): {pred_d} วัน | กทม. ซ่อมจริง: {actual_d} วัน (คลาดเคลื่อน: {abs(pred_d - actual_d):.2f} วัน)")
        print(f"[*] [Decision Layer]:")
        print(f"    - Public Impact Score : {impact_score} (ความหนาแน่นเขต: {triage['pop_density']} คน/ตร.กม.)")
        print(f"    - กลุ่มปฏิบัติการ กทม. : {category}")
        print(f"    - แผนการสั่งการ        : {action}")

        summary_rows.append({
            "Ticket ID": ticket_id,
            "เขต": district,
            "ประเภทหลัก": main_type,
            "งานย่อย (LLM)": sub_category,
            "ฝ่าย (LLM)": predicted_dept,
            "Severity": f"{severity}/5",
            "ทำนาย (วัน)": pred_d,
            "ซ่อมจริง (วัน)": actual_d,
            "Impact Score": impact_score,
            "กลุ่มปฏิบัติการ": category
        })

    # 5. สรุปผลตาราง
    print("\n" + "=" * 75)
    print("📊 ตารางสรุปผลลัพธ์ 11 Features End-to-End Pipeline (5 เคสจริง)")
    print("=" * 75)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
