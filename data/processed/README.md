# Cleaned Datasets (Traffy Fondue BMA)

ชุดข้อมูลที่ผ่านการทำความสะอาด (Cleaned) และสกัด Features สำหรับฝึกสอนโมเดล ML เรียบร้อยแล้ว:

## 1. รายการไฟล์ข้อมูล
- `train_cleaned.parquet` / `train_cleaned.csv`: ข้อมูลปี 2023 (177,726 เคส)
- `val_cleaned.parquet` / `val_cleaned.csv`: ข้อมูลต้นปี 2024 เดือน 1-3 (48,885 เคส)
- `test_cleaned.parquet` / `test_cleaned.csv`: ข้อมูลกลางปี 2024 เดือน 4-6 (49,010 เคส)

## 2. คอลัมน์สำคัญ (Features & Target)
- `duration_days`: Target ระยะเวลาซ่อมเสร็จจริง (วัน)
- `district`: เขต 50 เขตใน กทม.
- `main_type`: ประเภทปัญหาหลัก
- `sub_category`: เนื้องานย่อย
- `predicted_dept`: ฝ่ายที่รับผิดชอบ
- `severity`: ระดับความรุนแรง (1-5)
- `comment_len`: ความยาวข้อความร้องเรียน
- `day_of_week`: วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)
- `is_weekend`: วันหยุดสุดสัปดาห์ (0 หรือ 1)
- `month`: เดือนที่แจ้ง (1-12)
- `is_rainy_season`: ฤดูฝน พ.ค.-ต.ค. (0 หรือ 1)
- `hour`: เวลาที่แจ้ง (0-23 น.)
