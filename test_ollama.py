import sys
import json
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_comments = [
    "ฝาท่อแตกเปิดอ้าอยู่หน้าปากซอย คนเดินตกท่อไปแล้วเมื่อเช้า อันตรายถึงชีวิตมาก ช่วยด่วนครับ",
    "หลอดไฟทางดับ 1 ดวง ซอยค่อนข้างมืดช่วงกลางคืนครับ",
    "อยากให้มาตัดหญ้าข้างทางหน่อยครับ เริ่มยาวขึ้นมารกแล้ว"
]

def evaluate_with_ollama(comment, model="qwen3:4b"):
    prompt = f"""คุณคือ AI ผู้เชี่ยวชาญด้านการประเมินความปลอดภัยและจัดการปัญหาเมืองของ กทม.
หน้าที่ของคุณคืออ่าน "ข้อความร้องเรียนจากประชาชน" แล้วประเมินระดับความรุนแรง (Severity) เป็นตัวเลข 1 ถึง 5 ดังนี้:
1 = เล็กน้อย ไม่กระทบความปลอดภัย
2 = กวนใจเล็กน้อย ยังใช้งานได้ปกติ
3 = ปานกลาง รบกวนชีวิตประจำวัน (เช่น ไฟทางดับ, ทางเท้าแตก)
4 = สูง กระทบการสัญจรหนักหรือเสี่ยงอันตราย (เช่น ถนนเป็นหลุมลึก, ท่อระบายน้ำล้น)
5 = วิกฤต/อันตรายถึงชีวิต (เช่น ฝาท่อเปิดอ้าคนตกได้, เสาไฟจะล้ม, กลิ่นแก๊สรั่ว)

ข้อความร้องเรียน: "{comment}"

ตอบกลับเฉพาะ JSON รูปแบบนี้เท่านั้น:
{{"severity": <ตัวเลข 1-5>, "reason": "<เหตุผลสั้นๆ ไม่เกิน 1 ประโยค>"}}
"""
    payload = {
        "model": model,
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
            data = json.loads(resp.read().decode("utf-8"))
            raw_res = data.get("response", "")
            print("    [DEBUG RAW]:", repr(raw_res))
            # ค้นหา JSON ในสตริง
            import re
            match = re.search(r'\{.*?\}', raw_res, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw_res)
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 60)
    print("ทดสอบเรียกใช้ Ollama (Model: qwen3:4b) บนเครื่องคุณ")
    print("=" * 60)

    for idx, c in enumerate(test_comments, 1):
        print(f"\n[{idx}] ข้อความร้องเรียน: '{c}'")
        res = evaluate_with_ollama(c)
        if "error" in res:
            print(f"    [!] Error: {res['error']}")
        else:
            print(f"    -> Severity Score : {res.get('severity')} / 5")
            print(f"    -> เหตุผลจาก LLM  : {res.get('reason')}")

if __name__ == "__main__":
    main()
