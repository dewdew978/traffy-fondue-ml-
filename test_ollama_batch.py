import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.llm_severity import SeverityExtractor

extractor = SeverityExtractor(ollama_model='qwen3:4b')

samples = [
    "ฝาท่อเปิดอ้าอยู่หน้าปากซอย คนเดินตกท่อไปคนนึงแล้ว อันตรายถึงชีวิตมาก ช่วยด่วนครับ",
    "หลอดไฟส่องสว่างริมทางดับ 1 ดวง ซอยมืดมากเวลากลางคืน เดินลำบาก",
    "ต้นหญ้าและกิ่งไม้ข้างทางเริ่มขึ้นรกแล้ว อยากให้ กทม. ส่งคนมาตัดแต่งให้สวยงาม"
]

print("=" * 65)
print("TEST: OLLAMA LOCAL LLM (Model: qwen3:4b) กับข้อความจริง Traffy Fondue")
print("=" * 65)

for i, text in enumerate(samples, 1):
    res = extractor.evaluate_comment(text, use_ollama=True)
    print(f"[{i}] ข้อความ : \"{text}\"")
    print(f"    -> ระดับความรุนแรง (Severity) : {res['severity']} / 5")
    print(f"    -> เหตุผลจาก Ollama          : {res['reason']}")
    print(f"    -> Engine                   : {res['engine']}\n")
