"""
data_preprocessing.py
---------------------
ทำความสะอาดข้อมูล (Data Cleaning) และเตรียม 11 Features สำหรับ LightGBM
อิงตาม Schema จริงของ Traffy Fondue Open Data กทม.:
1. district: ชื่อเขต (50 เขต)
2. main_type: ประเภทปัญหาหลัก
3. sub_category: เนื้องานย่อย (สกัดจากแท็กหรือข้อความ)
4. predicted_dept: ฝ่ายรับผิดชอบ (สกัดจาก organization_action ในอดีต)
5. severity: ระดับความรุนแรง (1 - 5)
6. comment_len: ความยาวข้อความร้องเรียน
7. day_of_week: วันในสัปดาห์ (0 - 6)
8. is_weekend: วันเสาร์-อาทิตย์ (0 หรือ 1)
9. month: เดือนที่แจ้ง (1 - 12)
10. is_rainy_season: หน้าฝน พ.ค. - ต.ค. (0 หรือ 1)
11. hour: เวลาที่แจ้ง (0 - 23 นาฬิกา)

Target: duration_days (ระยะเวลาซ่อมเสร็จจริง)
"""

import os
import re
import pandas as pd
import numpy as np


def parse_type_hierarchy(type_val: str):
    """
    แยกประเภทหลัก (main_type) และเนื้องานย่อย (sub_category) จาก type
    เช่น 'สาธารณูปโภค -> ไฟฟ้า -> ขอไฟฟ้า' -> ('สาธารณูปโภค', 'ขอไฟฟ้า')
    """
    if not isinstance(type_val, str) or not type_val.strip():
        return 'ทั่วไป', 'ทั่วไป'
    parts = [p.strip() for p in type_val.split('->')]
    main_t = parts[0] if parts[0] else 'ทั่วไป'
    sub_t = parts[-1] if len(parts) > 1 and parts[-1] else main_t
    return main_t, sub_t


def parse_department_from_org(org_str: str) -> str:
    """
    สกัดฝ่ายรับผิดชอบหลักจากประวัติ organization_action
    เพื่อใช้เป็น Label/Feature ในการฝึกสอนโมเดล
    """
    if not isinstance(org_str, str) or not org_str.strip():
        return 'สำนักงานเขตทั่วไป'
    org = org_str.lower()
    if 'รักษาความสะอาด' in org:
        return 'ฝ่ายรักษาความสะอาดฯ'
    if 'โยธา' in org:
        return 'ฝ่ายโยธา'
    if 'เทศกิจ' in org:
        return 'ฝ่ายเทศกิจ'
    if 'สิ่งแวดล้อม' in org or 'สุขาภิบาล' in org:
        return 'ฝ่ายสิ่งแวดล้อมฯ'
    if 'ระบายน้ำ' in org:
        return 'สำนักการระบายน้ำ'
    if 'การไฟฟ้า' in org or 'กฟน' in org or 'mea' in org:
        return 'การไฟฟ้านครหลวง'
    if 'การประปา' in org or 'กปน' in org or 'mwa' in org:
        return 'การประปานครหลวง'
    if 'จราจร' in org:
        return 'สำนักการจราจรและขนส่ง'
    return 'สำนักงานเขตทั่วไป'


def estimate_severity_batch(comments: pd.Series) -> pd.Series:
    """
    ประเมินระดับความรุนแรง (1 - 5) สำหรับชุดข้อมูลขนาดใหญ่ด้วย Fast Heuristic
    """
    s_series = pd.Series(2, index=comments.index)
    c_str = comments.fillna('').astype(str).str.lower()

    # ระดับ 3: ปานกลาง
    mask_3 = c_str.str.contains('มืด|ไฟดับ|เหม็น|น้ำขัง|กระเบื้องแตก|ฝาท่อชำรุด|รบกวน|เดือดร้อน|กลิ่น', regex=True)
    s_series[mask_3] = 3

    # ระดับ 4: สูง
    mask_4 = c_str.str.contains('อันตรายมาก|หลุมลึก|ล้ม|ขวางถนน|น้ำท่วมสูง|สะดุดล้ม|มอเตอร์ไซค์ล้ม|สายไฟขาด|ด่วนมาก', regex=True)
    s_series[mask_4] = 4

    # ระดับ 5: วิกฤต
    mask_5 = c_str.str.contains('อันตรายถึงชีวิต|ตกท่อ|เสาไฟล้ม|ไฟไหม้|แก๊สรั่ว|ยุบตัว|ทรุดตัวลึก|ช็อต', regex=True)
    s_series[mask_5] = 5

    # ระดับ 1: ปัญหาเล็กน้อย/ความสวยงาม
    mask_1 = (c_str.str.len() > 0) & c_str.str.contains('สีลอก|ป้ายเอียง|ทาสี|หญ้าขึ้น|ไม่สวย', regex=True) & (~mask_3) & (~mask_4) & (~mask_5)
    s_series[mask_1] = 1

    return s_series


def clean_traffy_data(raw_csv_path: str, max_days_threshold: float = 60.0) -> pd.DataFrame:
    """
    โหลดและทำความสะอาดข้อมูล Traffy Fondue จากไฟล์ CSV ทางการ:
    - กรองเฉพาะสถานะ 'เสร็จสิ้น'
    - คำนวณ duration_days
    - สกัด 11 Features หน้างาน
    """
    use_cols = [
        'ticket_id', 'type', 'district', 'subdistrict',
        'timestamp', 'state', 'duration_minutes_total', 'comment',
        'organization_action'
    ]

    sample_df = pd.read_csv(raw_csv_path, nrows=2)
    available_cols = [c for c in use_cols if c in sample_df.columns]

    df = pd.read_csv(raw_csv_path, usecols=available_cols, low_memory=False)

    # 1. กรองเฉพาะเคสที่ 'เสร็จสิ้น'
    if 'state' in df.columns:
        df = df[df['state'].astype(str).str.contains('เสร็จ|finish|closed', case=False, na=False)].copy()

    # 2. คำนวณ duration_days จาก duration_minutes_total
    if 'duration_minutes_total' in df.columns:
        df['duration_days'] = pd.to_numeric(df['duration_minutes_total'], errors='coerce') / 1440.0
    else:
        df['created_at'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if 'last_activity' in df.columns:
            df['finished_at'] = pd.to_datetime(df['last_activity'], errors='coerce')
            df['duration_days'] = (df['finished_at'] - df['created_at']).dt.total_seconds() / 86400.0

    # 3. ตัดเคสที่เป็น Outliers หรือข้อมูลผิดปกติ
    # (น้อยกว่า 0.04 วัน = จนท. ปิดทันที, มากกว่า 60 วัน = เคสดองงานข้ามปีงบประมาณ)
    df = df[df['duration_days'].notna()].copy()
    df = df[(df['duration_days'] >= 0.04) & (df['duration_days'] <= max_days_threshold)].copy()

    # 4. จัดการ District
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.replace('เขต', '').str.strip()
        df = df[df['district'].notna() & (df['district'] != 'nan') & (df['district'] != '')].copy()

    # 5. สกัด main_type และ sub_category
    if 'type' in df.columns:
        parsed_types = df['type'].fillna('ทั่วไป').astype(str).apply(parse_type_hierarchy)
        df['main_type'] = [pt[0] for pt in parsed_types]
        df['sub_category'] = [pt[1] for pt in parsed_types]
    else:
        df['main_type'] = 'ทั่วไป'
        df['sub_category'] = 'ทั่วไป'

    # 6. สกัด predicted_dept จาก organization_action
    if 'organization_action' in df.columns:
        df['predicted_dept'] = df['organization_action'].apply(parse_department_from_org)
    else:
        df['predicted_dept'] = 'สำนักงานเขตทั่วไป'

    # 7. สกัด Temporal Features
    df['created_at'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['created_at'].notna()].copy()

    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['month'] = df['created_at'].dt.month
    df['hour'] = df['created_at'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rainy_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)

    # 8. สกัด Text Features และ Severity
    if 'comment' in df.columns:
        df['comment_len'] = df['comment'].fillna('').astype(str).str.len()
        df['severity'] = estimate_severity_batch(df['comment'])
    else:
        df['comment_len'] = 0
        df['severity'] = 2

    return df


if __name__ == "__main__":
    print("Pre-processing script updated with 11 front-end features.")
