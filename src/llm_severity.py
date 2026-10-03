"""
llm_severity.py
---------------
โมดูล LLM Complaint Analyzer สำหรับสกัด Features สำคัญจากข้อความร้องเรียน (comment):
1. severity: ระดับความรุนแรง (1 - 5)
2. sub_category: เนื้องานย่อย (เช่น ฝาท่อชำรุด, ถนนเป็นหลุมบ่อ, ไฟฟ้าดับ, ขยะตกค้าง)
3. predicted_dept: ฝ่ายที่ควรรับผิดชอบ (เช่น ฝ่ายโยธา, ฝ่ายรักษาความสะอาดฯ, ฝ่ายเทศกิจ)
4. reason: เหตุผลประกอบการวิเคราะห์

รองรับ 2 โหมด:
1. Ollama Mode: รันผ่าน Local LLM (qwen3:4b) บน GPU/เครื่องตนเอง ไม่ต้องต่อเน็ต
2. Fast Heuristic Mode: กฎสแกนคำสำคัญภาษาไทยความเร็วสูง (Rule-based) สำหรับงาน Batch ขนาดใหญ่หรือ Fallback
"""

import os
import re
import json
import urllib.request

VALID_DEPARTMENTS = [
    "ฝ่ายโยธา",
    "ฝ่ายรักษาความสะอาดฯ",
    "ฝ่ายเทศกิจ",
    "สำนักการระบายน้ำ",
    "ฝ่ายสิ่งแวดล้อมฯ",
    "การไฟฟ้านครหลวง",
    "การประปานครหลวง",
    "สำนักการจราจรและขนส่ง",
    "สำนักงานเขตทั่วไป"
]

SYSTEM_PROMPT = """คุณคือ AI ผู้เชี่ยวชาญด้านการจำแนกปัญหาเมืองและงานปฏิบัติการของกรุงเทพมหานคร (กทม.)
ให้อ่าน "ข้อความร้องเรียนจากประชาชน" แล้ววิเคราะห์ข้อมูล 4 อย่างและตอบกลับเป็น JSON ดังนี้:

1. severity: ตัวเลข 1 ถึง 5
   - 1: ปัญหาความสวยงาม ไม่กระทบความปลอดภัย เช่น สีสะพานลอก ป้ายเอียงเล็กน้อย
   - 2: กวนใจเล็กน้อย ยังใช้งานได้ปกติ เช่น หญ้าขึ้นรก มีเศษขยะเล็กน้อย
   - 3: ปานกลาง รบกวนชีวิตประจำวัน เช่น ไฟทางดับ 1 ดวง ทางเท้ามีน้ำขัง กระเบื้องแตกเดินสะดุด
   - 4: สูง เสี่ยงอันตราย/กระทบการจราจรวงกว้าง เช่น ถนนเป็นหลุมลึก ต้นไม้พาดสายไฟ ท่อระบายน้ำล้นผิวถนน
   - 5: วิกฤต/อันตรายถึงชีวิต เช่น ฝาท่อเปิดอ้า เสาไฟเอียงจะล้ม กลิ่นแก๊สรั่ว ถนนทรุดตัวลึก

2. sub_category: เนื้องานย่อยระบุชัดเจน (เช่น 'ฝาท่อชำรุด', 'ถนนเป็นหลุมบ่อ', 'ทางเท้าชำรุด', 'ไฟฟ้าส่องสว่างดับ', 'ขยะตกค้าง', 'ตัดแต่งต้นไม้', 'ตั้งวางกีดขวาง', 'ขุดลอกท่อ', 'ท่อประปาแตก')

3. predicted_dept: ฝ่ายรับผิดชอบหลักที่ตรงที่สุด เลือก 1 จากรายการนี้เท่านั้น:
   ['ฝ่ายโยธา', 'ฝ่ายรักษาความสะอาดฯ', 'ฝ่ายเทศกิจ', 'สำนักการระบายน้ำ', 'ฝ่ายสิ่งแวดล้อมฯ', 'การไฟฟ้านครหลวง', 'การประปานครหลวง', 'สำนักการจราจรและขนส่ง', 'สำนักงานเขตทั่วไป']

4. reason: เหตุผลสั้นๆ 1 ประโยค

ตอบกลับเฉพาะ JSON รูปแบบนี้เท่านั้น:
{"severity": <1-5>, "sub_category": "<เนื้องานย่อย>", "predicted_dept": "<ฝ่ายรับผิดชอบ>", "reason": "<เหตุผล>"}
"""


class LLMComplaintAnalyzer:
    def __init__(self, ollama_model: str = "qwen3:4b", ollama_host: str = "http://localhost:11434", timeout: float = 3.0):
        self.ollama_model = ollama_model
        self.ollama_url = f"{ollama_host}/api/generate"
        self.timeout = timeout

    def analyze(self, text: str) -> dict:
        """อินเทอร์เฟซหลักสำหรับการวิเคราะห์ข้อความ (Ollama + Auto Heuristic Fallback)"""
        return self.analyze_with_ollama(text)

    def analyze_with_ollama(self, text: str) -> dict:
        """
        วิเคราะห์ข้อความด้วย Ollama Local LLM
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "severity": 2,
                "sub_category": "ทั่วไป",
                "predicted_dept": "สำนักงานเขตทั่วไป",
                "reason": "ไม่มีข้อความบรรยาย",
                "engine": f"ollama ({self.ollama_model})"
            }

        prompt = f"""{SYSTEM_PROMPT}

ข้อความร้องเรียน: "{text.strip()}"

ตอบกลับเฉพาะ JSON เท่านั้น:"""

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_res = data.get("response", "")

                # สกัด JSON จาก response
                match = re.search(r'\{.*?\}', raw_res, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))

                    # ตรวจสอบความถูกต้องของ severity
                    sev = int(parsed.get("severity", 3))
                    sev = max(1, min(5, sev))

                    # ตรวจสอบ sub_category
                    sub_cat = str(parsed.get("sub_category", "ทั่วไป")).strip()
                    if not sub_cat:
                        sub_cat = "ทั่วไป"

                    # ตรวจสอบ predicted_dept ให้อยู่ในกลุ่มที่ถูกต้อง
                    dept = str(parsed.get("predicted_dept", "สำนักงานเขตทั่วไป")).strip()
                    if dept not in VALID_DEPARTMENTS:
                        # ค้นหาฝ่ายที่ใกล้เคียงที่สุด
                        matched_dept = "สำนักงานเขตทั่วไป"
                        for valid_d in VALID_DEPARTMENTS:
                            if valid_d in dept or dept in valid_d:
                                matched_dept = valid_d
                                break
                        dept = matched_dept

                    return {
                        "severity": sev,
                        "sub_category": sub_cat,
                        "predicted_dept": dept,
                        "reason": parsed.get("reason", "ประเมินโดย Ollama"),
                        "engine": f"ollama ({self.ollama_model})"
                    }
        except Exception:
            # Fallback หาก Ollama มีปัญหาหรือ timeout
            pass

        fallback = self.analyze_with_heuristic(text)
        fallback["engine"] = "heuristic_fallback"
        return fallback

    def analyze_with_heuristic(self, text: str) -> dict:
        """
        โหมดความเร็วสูง (Rule-based Fast Heuristic)
        สกัด severity, sub_category, และ predicted_dept จากคำสำคัญภาษาไทย
        """
        if not text or not isinstance(text, str):
            return {
                "severity": 2,
                "sub_category": "ทั่วไป",
                "predicted_dept": "สำนักงานเขตทั่วไป",
                "reason": "ไม่มีข้อความบรรยาย",
                "engine": "heuristic"
            }

        t = text.lower()

        # 1. Severity Rule-based
        severity = 2
        reason = "ข้อความทั่วไป ไม่พบคำเตือนวิกฤต"

        danger_l5 = ["อันตรายถึงชีวิต", "ตกท่อ", "เสาไฟล้ม", "ไฟไหม้", "แก๊สรั่ว", "ยุบตัว", "ทรุดตัวลึก", "ช็อต"]
        danger_l4 = ["อันตรายมาก", "หลุมลึก", "ล้ม", "ขวางถนน", "น้ำท่วมสูง", "สะดุดล้ม", "มอเตอร์ไซค์ล้ม", "สายไฟขาด", "ด่วนมาก"]
        danger_l3 = ["มืด", "ไฟดับ", "เหม็น", "น้ำขัง", "กระเบื้องแตก", "ฝาท่อชำรุด", "รบกวน", "เดือดร้อน", "กลิ่น"]
        danger_l2 = ["ขยะ", "หญ้า", "รก", "ป้าย", "ทาสี", "เสียงดัง"]

        if any(w in t for w in danger_l5):
            severity = 5
            reason = "พบความเสี่ยงอุบัติเหตุร้ายแรงหรืออันตรายถึงชีวิต"
        elif any(w in t for w in danger_l4):
            severity = 4
            reason = "พบคำระบุอุบัติเหตุหรือกีดขวางเส้นทางสัญจร"
        elif any(w in t for w in danger_l3):
            severity = 3
            reason = "กระทบการใช้ชีวิตประจำวันหรือสภาพแวดล้อม"
        elif any(w in t for w in danger_l2):
            severity = 2
            reason = "ปัญหาการบำรุงรักษาทั่วไป"

        # 2. Sub-category & Department Rule-based
        sub_category = "ทั่วไป"
        predicted_dept = "สำนักงานเขตทั่วไป"

        if "ฝาท่อ" in t:
            sub_category = "ฝาท่อชำรุด"
            predicted_dept = "สำนักการระบายน้ำ" if "ระบาย" in t else "ฝ่ายโยธา"
        elif "หลุม" in t or "แอสฟัลต์" in t or "ยางมะตอย" in t or "ถนนทรุด" in t:
            sub_category = "ถนนเป็นหลุมบ่อ"
            predicted_dept = "ฝ่ายโยธา"
        elif "ทางเท้า" in t or "กระเบื้อง" in t or "ฟุตบาท" in t:
            sub_category = "ทางเท้าชำรุด"
            predicted_dept = "ฝ่ายโยธา"
        elif "ไฟดับ" in t or "หลอดไฟ" in t or "เสาไฟ" in t or "ส่องสว่าง" in t or "มืด" in t:
            sub_category = "ไฟฟ้าส่องสว่างดับ"
            predicted_dept = "การไฟฟ้านครหลวง" if "เสาไฟ" in t or "สายไฟ" in t else "ฝ่ายโยธา"
        elif "ขยะ" in t or "ถังขยะ" in t or "ตกค้าง" in t:
            sub_category = "ขยะตกค้าง"
            predicted_dept = "ฝ่ายรักษาความสะอาดฯ"
        elif "ต้นไม้" in t or "กิ่งไม้" in t or "หญ้า" in t:
            sub_category = "ตัดแต่งต้นไม้"
            predicted_dept = "ฝ่ายรักษาความสะอาดฯ"
        elif "น้ำท่วม" in t or "ระบายน้ำ" in t or "ลอกท่อ" in t or "น้ำขัง" in t:
            sub_category = "ท่อระบายน้ำอุดตัน"
            predicted_dept = "สำนักการระบายน้ำ"
        elif "ประปา" in t or "น้ำไม่ไหล" in t or "ท่อแตก" in t:
            sub_category = "ท่อประปาแตก"
            predicted_dept = "การประปานครหลวง"
        elif "จอดรถ" in t or "กีดขวาง" in t or "หาบเร่" in t or "แผงลอย" in t:
            sub_category = "ตั้งวางกีดขวาง"
            predicted_dept = "ฝ่ายเทศกิจ"
        elif "สัญญาณไฟ" in t or "ทางม้าลาย" in t or "ป้ายจราจร" in t:
            sub_category = "อุปกรณ์จราจรชำรุด"
            predicted_dept = "สำนักการจราจรและขนส่ง"
        elif "กลิ่น" in t or "ควัน" in t or "เสียงดัง" in t or "ฝุ่น" in t or "pm2.5" in t:
            sub_category = "มลพิษสิ่งแวดล้อม"
            predicted_dept = "ฝ่ายสิ่งแวดล้อมฯ"

        return {
            "severity": severity,
            "sub_category": sub_category,
            "predicted_dept": predicted_dept,
            "reason": reason,
            "engine": "heuristic"
        }

    def evaluate_comment(self, text: str, use_ollama: bool = True) -> dict:
        if use_ollama:
            return self.analyze_with_ollama(text)
        return self.analyze_with_heuristic(text)


# สำหรับ Backward Compatibility
SeverityExtractor = LLMComplaintAnalyzer


if __name__ == "__main__":
    analyzer = LLMComplaintAnalyzer()
    test_comment = "ฝาท่อระบายน้ำแตกเปิดอ้าอยู่หน้าปากซอย คนเดินตกท่อไปแล้วเมื่อเช้า มอเตอร์ไซค์เกือบล้ม อันตรายมากครับ"
    print("Testing Comment:", test_comment)
    res = analyzer.evaluate_comment(test_comment, use_ollama=True)
    print("Result:", res)
