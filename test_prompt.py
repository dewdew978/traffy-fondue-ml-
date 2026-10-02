import sys
import io
import json
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

prompt = """คุณคือ AI ผู้เชี่ยวชาญด้านการจำแนกปัญหาเมืองและงานปฏิบัติการ กทม.
ให้อ่านข้อความร้องเรียนจากประชาชน แล้ววิเคราะห์เป็น JSON ดังนี้:
1. severity: ระดับความรุนแรง 1-5 (1=สวยงาม, 2=กวนใจเล็กน้อย, 3=ปานกลาง/รบกวนชีวิต, 4=อันตราย/เสี่ยงอุบัติเหตุ, 5=วิกฤต/อันตรายถึงชีวิต)
2. sub_category: เนื้องานย่อย (เช่น 'ฝาท่อชำรุด', 'ถนนเป็นหลุมบ่อ', 'ไฟฟ้าส่องสว่างดับ', 'ขยะตกค้าง', 'ตัดแต่งต้นไม้', 'ตั้งวางกีดขวาง', 'ขุดลอกท่อ', 'ท่อประปาแตก')
3. predicted_dept: ฝ่ายรับผิดชอบหลักที่ตรงที่สุด เลือกจาก: ['ฝ่ายโยธา', 'ฝ่ายรักษาความสะอาดฯ', 'ฝ่ายเทศกิจ', 'สำนักการระบายน้ำ', 'ฝ่ายสิ่งแวดล้อมฯ', 'การไฟฟ้านครหลวง', 'การประปานครหลวง', 'สำนักการจราจรและขนส่ง', 'สำนักงานเขตทั่วไป']
4. reason: เหตุผลสั้นๆ 1 ประโยค

ข้อความร้องเรียน: "ฝาท่อระบายน้ำแตกเปิดอ้าอยู่หน้าปากซอย คนเดินตกท่อไปแล้วเมื่อเช้า มอเตอร์ไซค์เกือบล้ม อันตรายมากครับ"
ตอบกลับเฉพาะ JSON เท่านั้น:
"""

payload = {
    "model": "qwen3:4b",
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("[Ollama Raw Response]:")
        print(res.get("response"))
except Exception as e:
    print(f"Error: {e}")
