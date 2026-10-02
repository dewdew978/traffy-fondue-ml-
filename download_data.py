"""
download_data.py
----------------
สคริปต์ดาวน์โหลดข้อมูล Traffy Fondue รายเดือนตาม "แนวทาง A: มาตรฐานสมบูรณ์แบบ":
- ปี 2023 ทั้งปี (2023-01 ถึง 2023-12) สำหรับ Train Set
- ปี 2024 ครึ่งปีแรก (2024-01 ถึง 2024-06) สำหรับ Validation & Test Set
"""

import sys
import os
import time
import urllib.request
import urllib.parse

# ตั้งค่า encoding สำหรับ console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# รายการไฟล์ที่ต้องการดาวน์โหลดตามแนวทาง A
TARGET_MONTHS = [
    # 2023 (Train Set ครบ 1 ปี)
    "bangkok_2023-01", "bangkok_2023-02", "bangkok_2023-03", "bangkok_2023-04",
    "bangkok_2023-05", "bangkok_2023-06", "bangkok_2023-07", "bangkok_2023-08",
    "bangkok_2023-09", "bangkok_2023-10", "bangkok_2023-11", "bangkok_2023-12",
    # 2024 (Validation & Test Set ครึ่งปีแรก)
    "bangkok_2024-01", "bangkok_2024-02", "bangkok_2024-03", "bangkok_2024-04",
    "bangkok_2024-05", "bangkok_2024-06"
]

BASE_API_URL = "https://publicapi.traffy.in.th/teamchadchart-stat-api/download/bangkok_monthly"

QUERY_PARAMS = {
    "email": "67070098@kmitl.ac.th",
    "name": "ปวริศ ปัญสิงห์",
    "org": "kmitl",
    "purpose": "for test",
    "tel": "0834388580"
}

def download_file(file_name, output_dir):
    out_path = os.path.join(output_dir, f"{file_name}.csv")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000000:
        print(f"[-] {file_name}.csv มีอยู่แล้ว ({os.path.getsize(out_path) / 1024 / 1024:.2f} MB) -> ข้ามการโหลด")
        return True

    params = QUERY_PARAMS.copy()
    params["file_name"] = file_name
    query_str = urllib.parse.urlencode(params)
    url = f"{BASE_API_URL}?{query_str}"

    print(f"[+] กำลังดาวน์โหลด: {file_name}.csv ...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        start_t = time.time()
        with urllib.request.urlopen(req, timeout=180) as response, open(out_path, 'wb') as out_file:
            total_bytes = 0
            while True:
                chunk = response.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                out_file.write(chunk)
                total_bytes += len(chunk)
        
        elapsed = time.time() - start_t
        mb_size = total_bytes / (1024 * 1024)
        print(f"    [OK] สำเร็จ: {mb_size:.2f} MB (ใช้เวลา {elapsed:.1f} วินาที)")
        return True
    except Exception as e:
        print(f"    [ERROR] เกิดข้อผิดพลาดในการโหลด {file_name}: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("[START] เริ่มต้นดาวน์โหลดข้อมูล Traffy Fondue (แนวทาง A: มาตรฐานสมบูรณ์แบบ)")
    print(f"Directory: {output_dir}")
    print(f"Total: {len(TARGET_MONTHS)} files (ปี 2023 ครบปี + ครึ่งปีแรก 2024)")
    print("=" * 65)

    success_count = 0
    for idx, month_file in enumerate(TARGET_MONTHS, 1):
        print(f"\n({idx}/{len(TARGET_MONTHS)})", end=" ")
        if download_file(month_file, output_dir):
            success_count += 1
        time.sleep(1)  # เว้นช่วง 1 วินาทีไม่ให้ยิง request ถี่เกินไป

    print("\n" + "=" * 65)
    print(f"[DONE] ดาวน์โหลดเสร็จสิ้น: สำเร็จ {success_count}/{len(TARGET_MONTHS)} ไฟล์")
    print(f"Folder: {output_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()
