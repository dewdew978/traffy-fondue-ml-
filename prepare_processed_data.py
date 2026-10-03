"""
prepare_processed_data.py
-------------------------
สคริปต์ประมวลผลและทำความสะอาดข้อมูล (Stage 1: Data Preparation Pipeline):
1. อ่านข้อมูลดิบจาก data/raw/
2. ผ่านขั้นตอน Data Cleaning, Outlier Filtering และ Feature Extraction (11 Features + Target)
3. บันทึกผลลัพธ์เป็นไฟล์ชุดข้อมูลที่คลีนสมบูรณ์ลงใน data/processed/:
   - train_cleaned.parquet & train_cleaned.csv (ปี 2023 ทั้งปี: 12 เดือน)
   - val_cleaned.parquet   & val_cleaned.csv   (ต้นปี 2024: เดือน 1-3)
   - test_cleaned.parquet  & test_cleaned.csv  (กลางปี 2024: เดือน 4-6)
"""

import sys
import os
import glob
import time
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data_preprocessing import clean_traffy_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_and_save(file_list, base_name, desc):
    print(f"\n[+] กำลังคลีนข้อมูลสำหรับ {desc} ({len(file_list)} ไฟล์)...")
    start_t = time.time()
    dfs = []

    for f in file_list:
        fname = os.path.basename(f)
        try:
            df = clean_traffy_data(f, max_days_threshold=60.0)
            dfs.append(df)
            print(f"    - {fname}: {len(df):,} เคสที่สมบูรณ์")
        except Exception as e:
            print(f"    [!] ผิดพลาดในการอ่าน {fname}: {e}")

    if not dfs:
        print(f"[!] ไม่มีข้อมูลสำหรับ {desc}")
        return None

    merged = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - start_t

    parquet_path = os.path.join(PROCESSED_DIR, f"{base_name}.parquet")
    csv_path = os.path.join(PROCESSED_DIR, f"{base_name}.csv")

    print(f"    --> กำลังบันทึกไฟล์สู่ data/processed/...")
    # บันทึกเป็น Parquet สำหรับโมเดลโหลดด้วยความเร็วสูง
    merged.to_parquet(parquet_path, index=False)
    # บันทึกเป็น CSV สำหรับเปิดอ่านใน Excel หรือ Text Editor
    merged.to_csv(csv_path, index=False, encoding='utf-8-sig')

    parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

    print(f"--> [สำเร็จ] รวม {len(merged):,} แถว ({elapsed:.1f} วินาที)")
    print(f"    - Parquet : {parquet_path} ({parquet_size_mb:.2f} MB)")
    print(f"    - CSV     : {csv_path} ({csv_size_mb:.2f} MB)")

    return merged


def main():
    print("=" * 75)
    print("เริ่มกระบวนการ Stage 1: Data Preprocessing & Export Cleaned Data")
    print("=" * 75)

    train_files = sorted(glob.glob(os.path.join(RAW_DIR, "bangkok_2023-*.csv")))
    val_files = [os.path.join(RAW_DIR, f"bangkok_2024-0{m}.csv") for m in [1, 2, 3]]
    test_files = [os.path.join(RAW_DIR, f"bangkok_2024-0{m}.csv") for m in [4, 5, 6]]

    # 1. Train Set ปี 2023
    process_and_save(train_files, "train_cleaned", "Train Set (ปี 2023 ทั้งปี)")

    # 2. Validation Set ต้นปี 2024
    process_and_save(val_files, "val_cleaned", "Validation Set (ต้นปี 2024: เดือน 1-3)")

    # 3. Test Set กลางปี 2024
    process_and_save(test_files, "test_cleaned", "Test Set (กลางปี 2024: เดือน 4-6)")

    # 4. เขียนคำอธิบายสรุป Metadata โฟลเดอร์ data/processed/
    readme_path = os.path.join(PROCESSED_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Cleaned Datasets (Traffy Fondue BMA)\n\n")
        f.write("ชุดข้อมูลที่ผ่านการทำความสะอาด (Cleaned) และสกัด Features สำหรับฝึกสอนโมเดล ML เรียบร้อยแล้ว:\n\n")
        f.write("## 1. รายการไฟล์ข้อมูล\n")
        f.write("- `train_cleaned.parquet` / `train_cleaned.csv`: ข้อมูลปี 2023 (177,726 เคส)\n")
        f.write("- `val_cleaned.parquet` / `val_cleaned.csv`: ข้อมูลต้นปี 2024 เดือน 1-3 (48,885 เคส)\n")
        f.write("- `test_cleaned.parquet` / `test_cleaned.csv`: ข้อมูลกลางปี 2024 เดือน 4-6 (49,010 เคส)\n\n")
        f.write("## 2. คอลัมน์สำคัญ (Features & Target)\n")
        f.write("- `duration_days`: Target ระยะเวลาซ่อมเสร็จจริง (วัน)\n")
        f.write("- `district`: เขต 50 เขตใน กทม.\n")
        f.write("- `main_type`: ประเภทปัญหาหลัก\n")
        f.write("- `sub_category`: เนื้องานย่อย\n")
        f.write("- `predicted_dept`: ฝ่ายที่รับผิดชอบ\n")
        f.write("- `severity`: ระดับความรุนแรง (1-5)\n")
        f.write("- `comment_len`: ความยาวข้อความร้องเรียน\n")
        f.write("- `day_of_week`: วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)\n")
        f.write("- `is_weekend`: วันหยุดสุดสัปดาห์ (0 หรือ 1)\n")
        f.write("- `month`: เดือนที่แจ้ง (1-12)\n")
        f.write("- `is_rainy_season`: ฤดูฝน พ.ค.-ต.ค. (0 หรือ 1)\n")
        f.write("- `hour`: เวลาที่แจ้ง (0-23 น.)\n")

    print("\n" + "=" * 75)
    print(f"[DONE] ส่งออกชุดข้อมูลที่คลีนแล้วสำเร็จทั้งหมดใน: {PROCESSED_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
