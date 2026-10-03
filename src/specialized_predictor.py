"""
specialized_predictor.py
------------------------
คลาสสำหรับการทำนายระยะเวลาซ่อมด้วยระบบ Specialized Sub-models (Mixture of Experts)
โดยใช้ predicted_dept เป็นตัว Router เลือกโมเดลเฉพาะทางของแต่ละฝ่าย:
1. ฝ่ายโยธา
2. ฝ่ายรักษาความสะอาดฯ
3. ฝ่ายเทศกิจ
4. ฝ่ายสิ่งแวดล้อมฯ
5. สำนักการระบายน้ำ
6. General Fallback Model (สำหรับฝ่ายอื่นๆ หรือเคสทั่วไป)
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "specialized")


DEPT_SLUG_MAP = {
    'ฝ่ายโยธา': 'yotha',
    'ฝ่ายรักษาความสะอาดฯ': 'cleanliness',
    'ฝ่ายเทศกิจ': 'thetsakit',
    'ฝ่ายสิ่งแวดล้อมฯ': 'environment',
    'สำนักการระบายน้ำ': 'drainage',
}


class SpecializedBMAPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.categories = {}
        self.dept_map = DEPT_SLUG_MAP.copy()
        self.feature_cols = [
            'district', 'main_type', 'sub_category',
            'severity', 'comment_len', 'day_of_week',
            'is_weekend', 'month', 'is_rainy_season', 'hour'
        ]
        self.cat_cols = ['district', 'main_type', 'sub_category']

        self._load_models()

    def _resolve_slug(self, dept_name: str) -> str:
        """แปลงชื่อฝ่ายภาษาไทยเป็น slug ของโมเดล"""
        dept_str = str(dept_name).strip()
        if dept_str in self.dept_map:
            slug = self.dept_map[dept_str]
        elif dept_str in self.models:
            slug = dept_str
        else:
            slug = "general"
        return slug if slug in self.models else "general"

    def _load_models(self):
        """โหลดโมเดลและ categories mapping ของแต่ละฝ่ายเข้าสู่หน่วยความจำ"""
        if not os.path.exists(self.models_dir):
            return

        dept_map_path = os.path.join(self.models_dir, "dept_map.json")
        if os.path.exists(dept_map_path):
            with open(dept_map_path, "r", encoding="utf-8") as f:
                self.dept_map.update(json.load(f))

        all_slugs = list(set(list(self.dept_map.values()) + ["general"]))
        for slug in all_slugs:
            model_path = os.path.join(self.models_dir, f"model_{slug}.txt")
            cat_path = os.path.join(self.models_dir, f"categories_{slug}.json")
            if os.path.exists(model_path):
                try:
                    self.models[slug] = lgb.Booster(model_file=model_path)
                    if os.path.exists(cat_path):
                        with open(cat_path, "r", encoding="utf-8") as f:
                            self.categories[slug] = json.load(f)
                except Exception as e:
                    print(f"[!] Warning: ไม่สามารถโหลดโมเดล {slug}: {e}")

    def is_ready(self) -> bool:
        return len(self.models) > 0

    def predict_single(self, feature_dict: dict, predicted_dept: str) -> dict:
        """
        ทำนายเคสเดี่ยว โดยเลือกโมเดลเฉพาะทางตาม predicted_dept
        """
        slug = self._resolve_slug(predicted_dept)
        model = self.models.get(slug) or self.models.get("general")

        if model is None:
            raise RuntimeError("ไม่มีโมเดลเฉพาะทางหรือ General Model อยู่ในระบบ")

        cats = self.categories.get(slug, {})
        df_row = pd.DataFrame([feature_dict])

        # จัดการ Categorical ให้ตรงกับโมเดล
        for c in self.cat_cols:
            known_cats = cats.get(c, [])
            val = df_row[c].iloc[0]
            if val not in known_cats:
                fallback_val = 'ทั่วไป' if 'ทั่วไป' in known_cats else (known_cats[0] if known_cats else val)
                df_row[c] = fallback_val
            df_row[c] = pd.Categorical(df_row[c], categories=known_cats)

        pred_log = model.predict(df_row[self.feature_cols])[0]
        pred_days = float(np.expm1(pred_log))

        return {
            "predicted_days": round(max(0.1, pred_days), 2),
            "model_used": f"Specialized ({predicted_dept})" if slug != "general" else "General Fallback",
            "routed_dept": predicted_dept,
            "model_slug": slug
        }

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ทำนายข้อมูลทั้ง DataFrame แบบแยกกลุ่มตาม predicted_dept อย่างรวดเร็ว (Vectorized)
        """
        out_df = df.copy()
        out_df['specialized_pred_days'] = 0.0
        out_df['model_used'] = ''

        dept_col = 'predicted_dept' if 'predicted_dept' in out_df.columns else None

        if dept_col is None:
            groups = [('general', out_df.index)]
        else:
            groups = [(str(k), idxs) for k, idxs in out_df.groupby(dept_col).groups.items()]

        for dept_str, idxs in groups:
            slug = self._resolve_slug(dept_str)
            model = self.models.get(slug) or self.models.get("general")
            if model is None:
                continue

            cats = self.categories.get(slug, {})
            sub_df = out_df.loc[idxs, self.feature_cols].copy()
            for c in self.cat_cols:
                known_cats = cats.get(c, [])
                sub_df[c] = pd.Categorical(sub_df[c], categories=known_cats)

            preds_log = model.predict(sub_df[self.feature_cols])
            preds_days = np.clip(np.expm1(preds_log), 0.1, None)

            out_df.loc[idxs, 'specialized_pred_days'] = np.round(preds_days, 2)
            model_label = f"Specialized ({dept_str})" if slug != "general" else "General Fallback"
            out_df.loc[idxs, 'model_used'] = model_label

        return out_df
