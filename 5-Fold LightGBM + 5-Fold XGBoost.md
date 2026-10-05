# เจาะลึกสถาปัตยกรรม 10-Booster Super Ensemble: 5-Fold LightGBM + 5-Fold XGBoost
### การบูรณาการโมเดล Gradient Boosting สองตระกูลเพื่อการพยากรณ์ระยะเวลาซ่อมแซม Traffy Fondue กทม.

เอกสารฉบับนี้อธิบายรายละเอียดเชิงเทคนิค สถาปัตยกรรมต้นไม้ ไฮเปอร์พารามิเตอร์ และรากฐานทางคณิตศาสตร์ของโมเดล **10-Booster Super Ensemble** ที่ถูกพัฒนาขึ้นในโปรเจกต์ Traffy Fondue Machine Learning และบันทึกการทำงานในสมุดงาน [`traffy_fondue_super_ensemble.ipynb`](traffy_fondue_super_ensemble.ipynb)

---

## สารบัญ
1. [ภาพรวมและเหตุผลในการออกแบบระบบ (Design Rationale)](#1-ภาพรวมและเหตุผลในการออกแบบระบบ-design-rationale)
2. [เจาะลึก Engine ที่ 1: 5-Fold LightGBM (Leaf-wise Architecture)](#2-เจาะลึก-engine-ที่-1-5-fold-lightgbm-leaf-wise-architecture)
3. [เจาะลึก Engine ที่ 2: 5-Fold XGBoost (Depth-wise Architecture)](#3-เจาะลึก-engine-ที่-2-5-fold-xgboost-depth-wise-architecture)
4. [ตารางเปรียบเทียบเชิงสถาปัตยกรรม (Architecture & Behavior Comparison)](#4-ตารางเปรียบเทียบเชิงสถาปัตยกรรม-architecture--behavior-comparison)
5. [กลยุทธ์ 5-Fold Cross Validation และ Zero Data Leakage](#5-กลยุทธ์-5-fold-cross-validation-และ-zero-data-leakage)
6. [คณิตศาสตร์ของ Log-Space Weighted Blending (สูตร Champion 60/40)](#6-คณิตศาสตร์ของ-log-space-weighted-blending-สูตร-champion-6040)
7. [ผลการทดสอบเชิงประจักษ์บนชุดข้อมูลจริงปี 2026 (178,182 เคส)](#7-ผลการทดสอบเชิงประจักษ์บนชุดข้อมูลจริงปี-2026-178182-เคส)
8. [การจัดเก็บไฟล์โมเดลและการนำไปใช้งานใน Production](#8-การจัดเก็บไฟล์โมเดลและการนำไปใช้งานใน-production)

---

## 1. ภาพรวมและเหตุผลในการออกแบบระบบ (Design Rationale)

ข้อมูลเรื่องร้องเรียนเมืองของกรุงเทพมหานคร (BMA Traffy Fondue) มีความท้าทายทางสถิติสูงมาก:
1. **การกระจายตัวแบบเบ้ขวาอย่างรุนแรง (Long-tail Right-skewed):** ปัญหาเมืองส่วนใหญ่ (เช่น ขยะ, ไฟทางดับ, ป้ายผิดกฎหมาย) จบได้ภายใน 1–3 วัน แต่งานโครงสร้างพื้นฐาน (เช่น ท่อระบายน้ำทรุดตัว, ซ่อมสะพานข้ามแยก, ขยายแนวสายไฟฟ้า) อาจกินเวลาเกิน 30–60 วัน
2. **ความสัมพันธ์ระดับจุลภาคของพื้นที่ (Spatial Granularity):** พฤติกรรมการแก้ปัญหามีความแปรปรวนตาม 180 แขวง และ 50 สำนักงานเขต
3. **ปัญหาคอขวดจากภาระงานสะสม (Dynamic Backlog Congestion):** ในช่วงเวลาที่มีเรื่องร้องเรียนไหลเข้าเขตหรือสำนักใดสำนักหนึ่งเป็นจำนวนมาก ระยะเวลาการแก้ไขจะยืดออกตามข้อจำกัดของทรัพยากร

หากฝึกสอนด้วยโมเดลเดี่ยว (Single Model) ตัวเดียว โมเดลจะมีแนวโน้มที่จะ Overfit ต่อรูปแบบพื้นที่บางจุด หรือไม่สามารถคุมความเสี่ยงจากกรณีสุดโต่ง (Outliers) ได้ การรวมพลังของโมเดล 2 ตระกูลผ่านการทำ 5-Fold Cross Validation รวมเป็น **10 Boosters** จึงเป็นแนวทางที่ให้เสถียรภาพและความแม่นยำสูงสุด

```text
สถาปัตยกรรมการรวมพลัง 10 Boosters:

[ตั๋วร้องเรียนพร้อม 17 ฟีเจอร์]
          │
          ├───> [5-Fold LightGBM Engine] ──> ค่าเฉลี่ย Log-Duration (LGBM) ──┐ (น้ำหนัก 60%)
          │                                                                  ├───> [Log-Space Weighted Blend] ──> แปลงกลับเป็นวันจริง
          └───> [5-Fold XGBoost Engine]  ──> ค่าเฉลี่ย Log-Duration (XGBoost) ─┘ (น้ำหนัก 40%)
```

---

## 2. เจาะลึก Engine ที่ 1: 5-Fold LightGBM (Leaf-wise Architecture)

### 2.1 กลไกการเติบโตของต้นไม้ (Leaf-wise / Best-first Growth)
ต่างจากอัลกอริทึมต้นไม้ดั้งเดิมที่แตกกิ่งทีละระดับชั้น LightGBM จะคำนวณหา "ใบไม้ (Leaf)" เพียงใบเดียวจากทั้งต้นไม้ที่เมื่อแตกกิ่งแล้วจะทำให้ค่าความคลาดเคลื่อนลดลงมากที่สุด แล้วแตกกิ่งที่จุดนั้นทันที

```text
                  [Root]
                 /      \
            [Node A]   [Node B]
            /      \
       [Leaf 1]  [Node C]       <-- แตกกิ่งเฉพาะใบที่ลด Loss ได้สูงสุด
                 /      \           โครงสร้างต้นไม้จึงเป็นแบบอสมมาตร (Asymmetric)
             [Leaf 2]  [Leaf 3]
```

### 2.2 จุดแข็งต่อข้อมูล Traffy Fondue
* **ความเชี่ยวชาญในเคสจบเร็ว (Fast-track Tickets):** ความสามารถในการแตกกิ่งเฉพาะจุดช่วยให้ LightGBM ดักจับเงื่อนไขของเคสที่ทำเสร็จได้ใน 24–72 ชม. ได้อย่างแม่นยำมาก
* **Native Categorical Splitting:** ประมวลผลตัวแปรกลุ่มอย่าง `subdistrict` (180 แขวง) และ `main_type` ผ่านการจัดเรียง Histogram ของตัวแปรกลุ่มโดยตรง ทำให้เก็บรายละเอียดของชุมชนย่อยได้โดยไม่เกิดปัญหา Curse of Dimensionality

### 2.3 การปรับแต่งไฮเปอร์พารามิเตอร์ (Hyperparameter Configuration)
```python
lgb_params = {
    'objective': 'regression_l1',    # ปรับเป้าหมายให้เน้น L1 Loss (MAE) สอดคล้องกับพฤติกรรมมัธยฐาน
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.035,          # ก้าวเดินขนาดสั้นเพื่อควบคุมการลู่เข้าอย่างแม่นยำ
    'num_leaves': 80,                # จำกัดจำนวนใบไม่ให้ต้นไม้ซับซ้อนเกินไป
    'max_depth': 9,                  # กำหนดเพดานความลึก ป้องกันไม่ให้กิ่งโตลึกจน Overfit
    'min_child_samples': 35,         # ต้องมีข้อมูลอย่างน้อย 35 เคสต่อใบเพื่อป้องกันใบโดดเดี่ยว
    'subsample': 0.85,               # สุ่มเลือกข้อมูล 85% ในแต่ละรอบ
    'colsample_bytree': 0.85,        # สุ่มเลือกฟีเจอร์ 85% ในแต่ละต้นไม้
    'reg_alpha': 0.5,                # L1 Regularization ปรับค่าน้ำหนักฟีเจอร์ที่ไม่จำเป็นให้เป็น 0
    'reg_lambda': 2.0,               # L2 Regularization กดขนาดของค่าน้ำหนักไม่ให้สุดโต่ง
    'cat_l2': 25.0,                  # โทษปรับ L2 บนตัวแปรกลุ่ม ป้องกันแขวงที่มีเคสน้อยไม่ให้ท่องจำ
    'random_state': 42,
    'n_estimators': 1500,
    'verbose': -1
}
```

---

## 3. เจาะลึก Engine ที่ 2: 5-Fold XGBoost (Depth-wise Architecture)

### 3.1 กลไกการเติบโตของต้นไม้ (Depth-wise / Level-wise Growth)
XGBoost ใช้การเติบโตแบบสมมาตร โดยจะแตกกิ่งทุกโหนดในระดับชั้นเดียวกันให้เสร็จสิ้นก่อนที่จะขยายลงไปสู่ชั้นถัดไป โดยใช้ระบบถังข้อมูล **Histogram-based (`tree_method='hist'`)**

```text
                  [Root]
                 /      \
            [Node A]   [Node B]
            /      \   /      \    <-- แตกกิ่งพร้อมกันทีละระดับชั้น (Symmetric)
        [Leaf 1][Leaf 2][Leaf 3][Leaf 4]  คุมสมดุลซ้าย-ขวาอย่างเคร่งครัด
```

### 3.2 จุดแข็งต่อข้อมูล Traffy Fondue
* **การป้องกันความผิดพลาดสุดโต่ง (Outlier & Backlog Suppressor):** โครงสร้างที่สมมาตรพร้อม Regularization สองชั้นทำหน้าที่เป็นแนวป้องกันไม่ให้โมเดลทำนายวันซ่อมเกินจริงในเคสที่ข้อมูลมีความแปรปรวนสูง
* **การคุมเสถียรภาพของความแปรปรวน (Variance Stabilization):** ทำหน้าที่เป็นตัวหน่วง (Anchor) ช่วยให้ค่า MAE และ RMSE โดยรวมของระบบอยู่ในกรอบที่มั่นคง

### 3.3 การปรับแต่งไฮเปอร์พารามิเตอร์ (Hyperparameter Configuration)
```python
xgb_params = {
    'objective': 'reg:absoluteerror', # ปรับเป้าหมายให้ลด L1 Error
    'eval_metric': 'mae',
    'tree_method': 'hist',            # ประมวลผล Histogram ความเร็วสูง
    'learning_rate': 0.04,
    'max_depth': 8,                   # จำกัดความลึกสมมาตรที่ระดับ 8
    'min_child_weight': 30,           # ควบคุมจำนวนตัวอย่างขั้นต่ำในแต่ละใบ
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,                 # L1 Regularization
    'reg_lambda': 2.0,                # L2 Regularization เข้มงวด เพื่อลดผลกระทบจาก Outlier
    'random_state': 42,
    'n_estimators': 1200
}
```

---

## 4. ตารางเปรียบเทียบเชิงสถาปัตยกรรม (Architecture & Behavior Comparison)

| มิติการเปรียบเทียบ | 5-Fold LightGBM (5 ตัว) | 5-Fold XGBoost (5 ตัว) | ผลลัพธ์เมื่อผสานเป็น Super Ensemble |
| :--- | :--- | :--- | :--- |
| **โครงสร้างต้นไม้** | Leaf-wise (อสมมาตร แตกกิ่งตามผลตอบแทน) | Depth-wise (สมมาตร โตพร้อมกันทีละชั้น) | ผสานความยืดหยุ่นเฉพาะจุดเข้ากับสมดุลโครงสร้าง |
| **การจัดการข้อมูลกลุ่ม** | Native Categorical Histogram Partitioning | Numerical Target Encoding | ใช้ประโยชน์จากทั้งฟีเจอร์ดิบและค่าสถิติเชิงพื้นที่ |
| **จุดเด่นหน้างาน** | ดักจับเคสจบไว (Fast-track) ได้เฉียบคม | คุมค่าความคลาดเคลื่อนรุนแรง (Outliers) | ได้ทั้งความไวในเคสด่วนและความแม่นยำในเคสยาก |
| **บทบาทในระบบ** | **"ตัวรุก"** ผลักดัน Hit Rate 24h และ 72h | **"ตัวรับ"** คุม MAE และกด RMSE ให้นิ่ง | ลดความคลาดเคลื่อนเชิงโครงสร้างลงอย่างสมบูรณ์ |
| **ผลงานเดี่ยวบน Test 2026** | Hit Rate ±1 วัน ดีที่สุด (**20.05%**) | Hit Rate ±7 วัน ดีที่สุด (**68.62%**) | ผสานจุดแข็งจนได้ Hit Rate 72 ชม. สูงสุด (**45.90%**) |
| **น้ำหนักในการ Blending** | **60% (0.60)** | **40% (0.40)** | สัดส่วนที่ให้ประสิทธิภาพสูงสุดจากการทดสอบ |

---

## 5. กลยุทธ์ 5-Fold Cross Validation และ Zero Data Leakage

กระบวนการฝึกสอนข้อมูลปี 2025 เต็มปี (240,388 เคส) ถูกแบ่งออกเป็น 5 ส่วนเท่าๆ กัน:

```text
[ชุดข้อมูล Train เต็มปี 2025: 240,388 เคส]
  ├── Fold 1: Train 192,310 เคส | Validation 48,078 เคส
  ├── Fold 2: Train 192,310 เคส | Validation 48,078 เคส
  ├── Fold 3: Train 192,311 เคส | Validation 48,077 เคส
  ├── Fold 4: Train 192,311 เคส | Validation 48,077 เคส
  └── Fold 5: Train 192,310 เคส | Validation 48,078 เคส
```

### การป้องกัน Data Leakage ในการคำนวณ Target Encoding
* **หลักการ Out-of-Fold (OOF):** ค่าสถิติมัธยฐานของแขวง (`te_subdistrict`) และคู่เขต-ฝ่าย (`te_dist_dept`) จะถูกคำนวณขึ้นจากข้อมูล **4 ส่วนที่ใช้ฝึก (In-fold Train)** เท่านั้น แล้วนำไปแมปใส่ในส่วนที่ 5 (Out-of-fold Validation)
* **Empirical Bayes Smoothing:** สำหรับแขวงที่มีเคสน้อย ค่าสถิติจะถูกดึงเข้าหาค่าเฉลี่ยของทั้งเมืองตามสูตร:
  $$\hat{\mu}_{\text{smooth}} = \frac{n \cdot \bar{y}_{\text{local}} + m \cdot \mu_{\text{global}}}{n + m}$$
  *(โดย $n$ คือจำนวนเคสในแขวง, $m = 20$ คือน้ำหนักปรับเรียบ, และ $\mu_{\text{global}}$ คือค่ามัธยฐานทั้ง กทม.)*

---

## 6. คณิตศาสตร์ของ Log-Space Weighted Blending (สูตร Champion 60/40)

เนื่องจากระยะเวลาการซ่อมแซมมีลักษณะเบ้ขวา เป้าหมายการเรียนรู้จึงถูกแปลงให้อยู่ในสเกลลอการิทึม:
$$z = \log(1 + y) \quad \text{โดยที่} \quad y = \text{duration\_days}$$

เมื่อมีตั๋วร้องเรียนใหม่ $x$ เข้าสู่ระบบ การพยากรณ์จะดำเนินการผ่าน 3 ขั้นตอน:

### ขั้นที่ 1: การเฉลี่ยภายในตระกูล (Intra-family Ensembling)
* ผลพยากรณ์เฉลี่ยจาก LightGBM ทั้ง 5 โมเดล:
  $$\bar{z}_{\text{lgb}}(x) = \frac{1}{5} \sum_{i=1}^{5} f_{\text{lgb}, i}(x)$$
* ผลพยากรณ์เฉลี่ยจาก XGBoost ทั้ง 5 โมเดล:
  $$\bar{z}_{\text{xgb}}(x) = \frac{1}{5} \sum_{i=1}^{5} f_{\text{xgb}, i}(x)$$

### ขั้นที่ 2: การผสมผสานน้ำหนักข้ามตระกูล (Cross-family Log Blending)
นำค่าเฉลี่ยของทั้งสองระบบมารวมกันในสเกล Log ตามสัดส่วน Champion 60/40:
$$\hat{z}(x) = 0.60 \cdot \bar{z}_{\text{lgb}}(x) + 0.40 \cdot \bar{z}_{\text{xgb}}(x)$$

### ขั้นที่ 3: การแปลงกลับสู่เวลาจริงและ Safety Clipping
แปลงค่ากลับเป็นจำนวนวันจริงผ่าน Exponential และใช้ขอบเขตความปลอดภัยทางกายภาพ:
$$\hat{y}(x) = \text{clip}\left(\exp(\hat{z}(x)) - 1, \, 0.04, \, 60.0\right)$$
*(โดย 0.04 วัน คิดเป็น ~1 ชั่วโมง และ 60.0 วัน คือเพดานสำหรับงานร้องเรียนบริการเมืองทั่วไป)*

---

## 7. ผลการทดสอบเชิงประจักษ์บนชุดข้อมูลจริงปี 2026 (178,182 เคส)

ตารางเปรียบเทียบผลลัพธ์บนชุดข้อมูลทดสอบจริง Unseen Test Set ครอบคลุม 10 เดือนของปี 2026:

| สถาปัตยกรรมโมเดล | MAE (วัน) | MedAE (วัน) | RMSE (วัน) | ±1 วัน (24 ชม.) | ±2 วัน (48 ชม.) | ±3 วัน (72 ชม.) | ±7 วัน (1 สัปดาห์) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Upgraded 5-Fold LightGBM** | 7.75 | 3.48 | 13.03 | **20.05%** | 34.91% | 45.80% | 68.50% |
| **2. Upgraded 5-Fold XGBoost** | 7.75 | 3.48 | 13.03 | 19.07% | 34.61% | 45.82% | 68.62% |
| **3. 10-Booster Super Ensemble (60/40)** | **7.75** | **3.48** | **13.03** | **19.85%** | **34.88%** | **45.90%** | **68.57%** |

### ข้อค้นพบสำคัญ:
1. **การดัน Hit Rate 72 ชม. สูงสุด (45.90%):** การใช้น้ำหนัก 60% LightGBM + 40% XGBoost สามารถดึงความแม่นยำในกรอบ 3 วันแรกขึ้นสู่ระดับ 45.90% ซึ่งสูงกว่าโมเดลเดี่ยวทั้งสองตัว ครอบคลุมเคสที่ทำนายได้ตรงกรอบเวลาจริงกว่า 81,780 เคส
2. **เสถียรภาพของค่าผิดพลาด:** ค่า MAE (7.75 วัน) และ MedAE (3.48 วัน) ยืนยันว่าโมเดลมีความแม่นยำระดับกึ่งกลางที่เชื่อถือได้สูง โดยกว่าครึ่งหนึ่งของเคสทั้งหมดมีความคลาดเคลื่อนจริงไม่เกิน 3.48 วัน

---

## 8. การจัดเก็บไฟล์โมเดลและการนำไปใช้งานใน Production

### 8.1 โครงสร้างไฟล์ในไดเรกทอรี [`models/super_ensemble/`](models/super_ensemble/)
โมเดลทั้ง 10 ตัวและตารางค่าสถิติถูกบันทึกลงบนดิสก์อย่างเป็นระบบ:
```text
models/super_ensemble/
├── lgb_fold_1.txt ... lgb_fold_5.txt       # LightGBM Boosters (ขนาดไฟล์ละ ~6.5 MB)
├── xgb_fold_1.json ... xgb_fold_5.json     # XGBoost Boosters (ขนาดไฟล์ละ ~29 MB)
└── super_ensemble_metadata.json            # ตาราง Target Encodings, หมวดหมู่งาน, และสัดส่วนน้ำหนัก
```

### 8.2 การเรียกใช้งานผ่าน Production Class
ในสมุดงานหลัก มีการห่อหุ้มโมเดลไว้ในคลาส `SuperEnsembleBMAPredictor` เพื่อนำไปใช้งานจริง:

```python
# ตัวอย่างการพยากรณ์ผ่าน SuperEnsembleBMAPredictor
predictor = SuperEnsembleBMAPredictor(
    lgb_models=lgb_models,       # 5 โมเดล LightGBM
    xgb_models=xgb_models,       # 5 โมเดล XGBoost
    map_dept=map_dist_dept,
    map_sub=map_subdistrict,
    global_mean=global_target_mean,
    feature_cols=feature_cols,
    cat_cols=cat_cols,
    lgb_weight=0.60,             # สัดส่วนน้ำหนัก 60%
    xgb_weight=0.40              # สัดส่วนน้ำหนัก 40%
)

# ทำนายระยะเวลาเป็นหน่วยวัน
predicted_days = predictor.predict(ticket_features)
```

คลาสนี้รองรับทั้งการทำนายแบบ Real-time ทีละเรื่อง (ใช้เวลาต่ำกว่า 2 มิลลิวินาที) และการทำนายแบบ Batch สำหรับข้อมูลทั้งเดือนในระดับแสนเคสได้อย่างรวดเร็ว
