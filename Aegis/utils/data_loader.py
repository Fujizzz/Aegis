# utils/data_loader.py
from typing import Dict, List
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from utils.label_schema import LabelSchema
from utils.text_input import build_news_text, tokenize_texts
from utils.predictions import sample_identity
from utils.run_archive import sha256
from utils.pseudo_policy import RELEASED_QUALITY_STATUSES


def optional_supervision(value, key):
    """Blank/None/NaN means missing; malformed or out-of-range data fails closed."""
    if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0 or (key.endswith('_label') and value > 1):
        raise ValueError(f"{key} must be finite and {'in [0, 1]' if key.endswith('_label') else 'nonnegative'}")
    return value


class NewsDataset(Dataset):
    """
    读取带伪标签的数据集。
    CSV 必含 label 和 text/content；title/subject/date 可选。
    伪标签（若存在）列名：
      - emotion_label, exaggeration_label, bias_label, commonsense_label
      - emotion_weight, exaggeration_weight, bias_weight, commonsense_weight  （可选）
    * 注意：验证集不使用伪标签；训练集缺失的伪标签会在 collate_fn 中用 mask=0 忽略。
    """
    def __init__(
        self,
        csv_file: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 512,
        mode: str = "train",          # 'train' or 'val'
        use_pseudo: bool = True,      # 仅在 mode=='train' 时生效
        *,
        label_schema: LabelSchema,
    ):
        super().__init__()
        # Preserve literal strings (e.g. "nan", "NA", zero-padded dates).
        # Empty cells are handled by the shared text/supervision normalizers.
        self.df = pd.read_csv(csv_file, keep_default_na=False,
                              dtype={key: str for key in ('title', 'text', 'content', 'subject', 'date', 'sample_id', 'event_name', 'event')})
        self.input_sha256 = sha256(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.use_pseudo = (mode == "train") and use_pseudo

        # Frozen releases distinguish safe reuse from missing or stale supervision.
        if self.use_pseudo and 'pseudo_status' in self.df.columns:
            pending = ~self.df['pseudo_status'].isin(
                ['reused_identical_normalized_text', 'generated_strict_json_v1'])
            if 'pseudo_quality_status' in self.df.columns:
                pending |= ~self.df['pseudo_quality_status'].isin(RELEASED_QUALITY_STATUSES)
            else:
                # New generated data must carry an explicit release-policy marker.
                pending |= self.df['pseudo_status'] == 'generated_strict_json_v1'
            if pending.any():
                raise ValueError(
                    f'{csv_file}: {int(pending.sum())} samples need verified dimension labels. '
                    'Complete the pseudo_regeneration_queue in a new data version, or explicitly '
                    'set use_pseudo_labels=false for a no-dimension-supervision experiment.'
                )

        # 基础列检查
        required = ["label"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV缺少必要列: {missing}")
        if not {'text', 'content'} & set(self.df.columns):
            raise ValueError('CSV requires text or content')

        # Canonical model targets are computed in memory; never overwrite source CSVs.
        self.label_schema = label_schema
        targets = []
        for row_index, value in enumerate(self.df["label"], start=2):
            try:
                targets.append(label_schema.encode_source(value))
            except ValueError as exc:
                raise ValueError(f"{csv_file}: CSV row {row_index}: {exc}") from exc
        self.df["label"] = targets

        # 记录伪标签相关列（是否真的存在由 __getitem__/collate_fn 决定）
        self.dim_names = ["emotion", "exaggeration", "bias", "commonsense"]

    def __len__(self) -> int:
        return len(self.df)

    def _build_text(self, row: pd.Series) -> str:
        return build_news_text(row)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        item: Dict = {
            "input_text": self._build_text(row),
            "label": int(row["label"]),
        }
        sample_id, identity_source = sample_identity(row, idx, self.input_sha256)
        item['metadata'] = dict(sample_id=sample_id, identity_source=identity_source,
                                event=str(row.get('event_name', '') or row.get('event', '')).strip())

        # 训练模式下，如存在伪标签/权重则携带；缺失以 None 返回，collate_fn 中会打 mask=0
        if self.use_pseudo:
            for d in self.dim_names:
                lab_col = f"{d}_label"
                w_col   = f"{d}_weight"
                if lab_col in self.df.columns:
                    val = row.get(lab_col)
                    item[lab_col] = optional_supervision(val, lab_col)
                if w_col in self.df.columns:
                    valw = row.get(w_col)
                    item[w_col] = optional_supervision(valw, w_col)

        return item


def collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer, max_len: int) -> Dict:
    """
    将样本打包为张量：
      - 必含：input_ids, attention_mask, label
      - 若训练样本中含有 *_label / *_weight，则额外返回：
          *_label:  [B] float
          *_mask:   [B] float (1 有监督, 0 缺失)
          *_weight: [B] float (如有；缺失权重默认 1，显式 0 屏蔽监督)
          *_wmask:  [B] float (仅记录权重是否提供，不再作为监督掩码)
      缺失标签始终 mask=0；整批缺失标签时省略该维度键。
    """
    texts = [b["input_text"] for b in batch]
    enc = tokenize_texts(tokenizer, texts, max_len)

    out: Dict = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
    if any('metadata' in b for b in batch):
        if not all('metadata' in b for b in batch):
            raise ValueError('Partial sample identity metadata')
        out['metadata'] = [b['metadata'] for b in batch]

    # 打包可选伪标签/权重（若不存在则不返回对应键）
    def pack_optional(key: str, default_fill: float = 0.0, make_mask_key: str = ""):
        vals = []
        has_any = False
        for b in batch:
            value = optional_supervision(b.get(key), key)
            if value is not None:
                vals.append(value)
                has_any = True
            else:
                vals.append(None)
        if not has_any:
            return  # 所有样本均无该键，直接跳过
        mask = torch.tensor([0.0 if v is None else 1.0 for v in vals], dtype=torch.float32)
        filled = torch.tensor([default_fill if v is None else float(v) for v in vals], dtype=torch.float32)
        out[key] = filled
        if make_mask_key:
            out[make_mask_key] = mask

    dims = ["emotion", "exaggeration", "bias", "commonsense"]
    for d in dims:
        pack_optional(f"{d}_label", default_fill=0.0, make_mask_key=f"{d}_mask")
    for d in dims:
        pack_optional(f"{d}_weight", default_fill=1.0, make_mask_key=f"{d}_wmask")

    return out


def build_dataset(
    csv_file: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int = 512,
    use_pseudo: bool = True,
    mode: str = "train",
    *,
    label_schema: LabelSchema,
) -> Dataset:
    return NewsDataset(
        csv_file=csv_file,
        tokenizer=tokenizer,
        max_len=max_len,
        mode=mode,
        use_pseudo=use_pseudo,
        label_schema=label_schema,
    )
