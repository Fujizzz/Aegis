"""Convert source event CSVs to the shared news input schema."""
import os
from decimal import Decimal, InvalidOperation
import numpy as np
import pandas as pd
from utils.label_schema import get_label_schema

def format_one_event_csv(in_csv: str, out_csv: str, event_name: str) -> pd.DataFrame:
    # ✅ 读取时把 id 当“普通字符串”，尽量避免被 pandas 读成 float
    # 比 "string" 更硬：str
    df = pd.read_csv(in_csv, dtype={"id": str} if "id" else None)

    # 必需列检查
    if "content" not in df.columns:
        raise ValueError(f"{event_name}: missing column 'content' in {in_csv}")
    if "label" not in df.columns:
        raise ValueError(f"{event_name}: missing column 'label' in {in_csv}")

    # ========== label 清洗：允许存在 NaN/字符串 ==========
    label_raw = df["label"].replace([np.inf, -np.inf], np.nan)

    # One schema for numeric and named labels, independent of pandas string dtype.
    # Non-empty invalid labels fail instead of being silently coerced or truncated.
    schema = get_label_schema("pheme_rumour_v1")
    df["label"] = label_raw.map(lambda value: np.nan if pd.isna(value) else schema.source_code(value))

    # 丢掉 label 为空的行（否则无法转 int）
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[warn] {event_name}: dropped {dropped} rows due to NA/invalid label")

    # 转 int 并校验只能是 0/1
    # Reject fractional/out-of-range labels instead of truncating them to integers.
    df["label"] = df["label"].map(schema.source_code)
    bad = sorted(set(df["label"].unique().tolist()) - {0, 1})
    if bad:
        raise ValueError(f"{event_name}: label must be 0/1, got {bad}. Please check {in_csv}")

    # ========== id 清洗：避免科学计数法、去掉 .0 ==========
    if "id" in df.columns:
        def _normalize_id(x) -> str:
            if pd.isna(x):
                return ""
            s = str(x).strip()

            # 去掉末尾 ".0"
            s = re.sub(r"\.0$", "", s)

            # 如果形如 5.53e+17（科学计数法文本），尝试转成完整整数串
            # 注意：若原始文件保存时已丢精度，这里无法恢复原始 id，只能规范化当前值
            if "e" in s.lower():
                try:
                    # Decimal 能稳定处理科学计数法字符串
                    n = Decimal(s)
                    # 转成整数文本（不带科学计数法）
                    s2 = format(n, "f")
                    # 去掉小数点及后面的 0（如果有）
                    if "." in s2:
                        s2 = s2.split(".")[0]
                    return s2
                except (InvalidOperation, ValueError):
                    return s
            return s

        import re
        df["id"] = df["id"].apply(_normalize_id)

    # ========== 输出格式化表 ==========
    out = pd.DataFrame({
        "title": "",
        "text": df["content"].fillna("").astype(str),
        "subject": "",
        "date": "",
        "label": df["label"].astype(int),

        # 额外字段保留
        "id": df["id"] if "id" in df.columns else "",
        "event_label": df["event_label"] if "event_label" in df.columns else "",
        "if_marked_label": df["if_marked_label"] if "if_marked_label" in df.columns else "",
        "event_name": event_name,
    })

    # 丢弃空文本
    out = out[out["text"].str.len() > 0].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[format] {event_name}: {len(out)} rows -> {out_csv}")
    return out
