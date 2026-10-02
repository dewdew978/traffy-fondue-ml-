"""
decision_layer.py
-----------------
Actionable Decision Layer (Post-Processing Layer)
คำนวณ Public Impact Score และจัดกลุ่ม Action Triage Matrix สำหรับ กทม.
"""

import numpy as np
import pandas as pd


class BMADecisionLayer:
    """
    คลาสสำหรับประเมินผลกระทบต่อสังคม (Public Impact Score)
    และจัดกลุ่มงานตาม Action Triage Matrix
    """

    def __init__(self, density_csv_path: str = None):
        if density_csv_path:
            self.density_df = pd.read_csv(density_csv_path)
            self.min_density = float(self.density_df['pop_density'].min())
            self.max_density = float(self.density_df['pop_density'].max())
            self.density_map = dict(zip(self.density_df['district_th'], self.density_df['pop_density']))
        else:
            # ค่าโดยประมาณของ กทม. (หนองจอก ~770 ถึง ป้อมปราบฯ ~20,700 คน/ตร.กม.)
            self.min_density = 770.0
            self.max_density = 20750.0
            self.density_map = {}

    def get_density_by_district(self, district: str) -> float:
        """ดึงค่าความหนาแน่นประชากรจากชื่อเขต (ตัดคำว่า 'เขต' ออกถ้ามี)"""
        clean_name = district.replace("เขต", "").strip() if isinstance(district, str) else ""
        return self.density_map.get(clean_name, 5000.0)  # Default fallback ค่าเฉลี่ย

    def calculate_density_factor(self, density: float) -> float:
        """
        แปลงค่าความหนาแน่นประชากรให้อยู่ในช่วง 0.1 - 1.0
        Density Factor = 0.1 + 0.9 * (Density - Min) / (Max - Min)
        """
        norm = (density - self.min_density) / (self.max_density - self.min_density + 1e-6)
        scaled = 0.1 + 0.9 * norm
        return float(np.clip(scaled, 0.1, 1.0))

    def evaluate_ticket(self, severity: int, district: str, predicted_days: float) -> dict:
        """
        ประเมินเคสข้อร้องเรียน:
        - Severity: 1 - 5 (จากข้อความร้องเรียน)
        - District: ชื่อเขต
        - Predicted Days: วันที่โมเดล LightGBM ทำนายได้
        """
        pop_density = self.get_density_by_district(district)
        df_factor = self.calculate_density_factor(pop_density)

        # คำนวณ Public Impact Score
        impact_score = severity * df_factor * max(predicted_days, 0.1)

        # จัดกลุ่มตาม Action Triage Matrix
        # Threshold: Impact Score Cutoff = 8.0, วันที่วิกฤต = 4.0 วัน
        is_high_impact = impact_score >= 8.0

        if is_high_impact and predicted_days > 4.0:
            category = "Critical Urgent (Priority 1)"
            action = "ระดมช่างข้ามเขต เปิดหน้างานเร่งด่วน 24 ชม. ไม่ให้ปัญหาลากยาว"
            priority = 1
        elif is_high_impact and predicted_days <= 4.0:
            category = "Quick Win"
            action = "ส่งหน่วยเคลื่อนที่เร็วเข้าเคลียร์ทันที จบงานไว ลดคนเดือดร้อนสะสม"
            priority = 2
        elif not is_high_impact and predicted_days > 4.0:
            category = "Scheduled Project"
            action = "เข้าแผนซ่อมบำรุงตามรอบงบประมาณ พร้อมขึ้นป้ายแจ้งกำหนดการล่วงหน้า"
            priority = 3
        else:
            category = "Routine Maintenance"
            action = "ส่งเข้าคิวงานซ่อมบำรุงประจำวันตามรอบปกติของฝ่ายโยธาประจำเขต"
            priority = 4

        return {
            "district": district,
            "pop_density": round(pop_density, 1),
            "density_factor": round(df_factor, 3),
            "severity": severity,
            "predicted_days": round(predicted_days, 2),
            "public_impact_score": round(impact_score, 2),
            "category": category,
            "priority_level": priority,
            "action_recommendation": action
        }


if __name__ == "__main__":
    # ทดสอบการทำงาน
    engine = BMADecisionLayer()
    sample_a = engine.evaluate_ticket(severity=3, district="หนองจอก", predicted_days=3.0)
    sample_b = engine.evaluate_ticket(severity=4, district="วัฒนา", predicted_days=5.2)

    print("--- ผลการทดสอบเคสจำลอง ---")
    print(f"เคส A (หนองจอก): Score = {sample_a['public_impact_score']} -> {sample_a['category']}")
    print(f"เคส B (วัฒนา):   Score = {sample_b['public_impact_score']} -> {sample_b['category']}")
