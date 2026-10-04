import argparse
import csv
from typing import Dict, List

from utils.logger import init_logging

from detector.first_layer_detector import FirstLayerDetector
from detector.second_layer_detector import SecondLayerDetector
from reporter.report_generator import ReportGenerator
from pipeline import detect_item
from probability import thresholds


def _read_dataset(csv_path: str) -> List[Dict]:
    """
    读取数据集，要求列：title, text, subject, date, label
    label: 真=1, 假=0
    若无 id 列则按行号生成。
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=1):
            rows.append({
                "id": r.get("id") or r.get("ID") or str(i),
                "title": r.get("title") or r.get("Title") or "",
                "content": r.get("text") or r.get("Text") or "",
                "subject": r.get("subject") or r.get("Subject") or "",
                "date": r.get("date") or r.get("Date") or "",
                "label": r.get("label") or r.get("Label")  # 1/0
            })
    return rows


def _normalize_actual_label(x: str) -> str:
    """把实际标签规范为 'real' 或 'fake'。1/true->real, 0/false->fake"""
    s = "" if x is None else str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "real"):
        return "real"
    if s in ("0", "false", "f", "no", "n", "fake"):
        return "fake"
    # 未知标签不能作为评测真值。
    raise ValueError("Unknown actual label")


def build_first_layer_aspects() -> Dict[str, str]:
    """
    第一层四个维度的打分提示词模板（输出仅为0~1小数，越高越可疑）。
    模板可使用 {title} {subject} {date} {content}
    """
    # return {
    #     "情绪煽动": (
    #         "仅输出0~1小数（越高越可疑）。评估新闻的情绪煽动/极化语气：\n"
    #         "标题:{title}\n主题:{subject}\n日期:{date}\n正文:\n{content}"
    #     ),
    #     "夸张程度": (
    #         "仅输出0~1小数（越高越可疑）。评估新闻的夸张/绝对化/标题党程度：\n"
    #         "标题:{title}\n主题:{subject}\n日期:{date}\n正文:\n{content}"
    #     ),
    #     "偏见倾向": (
    #         "仅输出0~1小数（越高越可疑）。评估新闻的立场偏见/单边叙事：\n"
    #         "标题:{title}\n主题:{subject}\n日期:{date}\n正文:\n{content}"
    #     ),
    #     "常识冲突": (
    #         "仅输出0~1小数（越高越可疑）。评估新闻与常识/已知事实的潜在冲突：\n"
    #         "标题:{title}\n主题:{subject}\n日期:{date}\n正文:\n{content}"
    #     ),
    # }

    return {
        "Emotion/Sentiment": (
            "Output only a decimal number between 0 and 1 (the higher, the more suspicious). "
            "Evaluate whether the news contains emotional or inflammatory/polarizing language:\n"
            "Title: {title}\nSubject: {subject}\nDate: {date}\nContent:\n{content}"
        ),
        "Exaggeration": (
            "Output only a decimal number between 0 and 1 (the higher, the more suspicious). "
            "Evaluate the degree of exaggeration, absolutist expressions, or sensational/clickbait style:\n"
            "Title: {title}\nSubject: {subject}\nDate: {date}\nContent:\n{content}"
        ),
        "Bias": (
            "Output only a decimal number between 0 and 1 (the higher, the more suspicious). "
            "Evaluate whether the news shows bias, one-sided narrative, or partisan stance:\n"
            "Title: {title}\nSubject: {subject}\nDate: {date}\nContent:\n{content}"
        ),
        "Commonsense Conflict": (
            "Output only a decimal number between 0 and 1 (the higher, the more suspicious). "
            "Evaluate whether the news contradicts commonsense or widely known facts:\n"
            "Title: {title}\nSubject: {subject}\nDate: {date}\nContent:\n{content}"
        ),
    }


def main():
    from utils.config import load_config
    parser = argparse.ArgumentParser(description="Multi-Agent Fake News Detector (Two-Layer, DeepSeek)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # 1) 配置与日志
    cfg = load_config(args.config)
    init_logging(cfg.get("logging", {}).get("level", "INFO"))

    # DeepSeek/网络参数
    import os
    api_key = os.environ.get(cfg["deepseek"].get("api_key_env", "DEEPSEEK_API_KEY"), "").strip()
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY in the environment")
    base_url = cfg["deepseek"].get("base_url", "https://api.deepseek.com")
    timeout = float(cfg["deepseek"].get("timeout", 45))
    proxy = cfg["deepseek"].get("proxy", "")
    ca_bundle = cfg["deepseek"].get("ca_bundle", "")

    # 模型名
    model_first = cfg.get("models", {}).get("first_layer", "deepseek-chat")
    model_second = cfg.get("models", {}).get("second_layer", "deepseek-chat")

    # 阈值与输出
    fake_th = float(cfg["detection"].get("fake_threshold", 0.7))
    real_th = float(cfg["detection"].get("real_threshold", 0.3))
    output_csv = cfg["detection"]["output_csv"]

    # 2) 读取数据
    data_file = cfg["data"]["input_file"]
    items = _read_dataset(data_file)

    # 3) 初始化第一层
    aspects = build_first_layer_aspects()

    # （可选）维度权重与自一致设置（若没有就用默认）
    weights = cfg.get("first_layer", {}).get("weights", {
        "常识冲突": 0.35, "夸张程度": 0.25, "偏见倾向": 0.20, "情绪煽动": 0.20
    })
    self_consistency_n = cfg.get("first_layer", {}).get("self_consistency_n", 1)
    prompt_variants_map = cfg.get("first_layer", {}).get("prompt_variants_map", None)

    first_layer = FirstLayerDetector(
        api_key=api_key,
        aspects=aspects,
        weights=weights,
        fake_threshold=fake_th,
        real_threshold=real_th,
        base_url=base_url,
        timeout=timeout,
        proxy=proxy,
        ca_bundle=ca_bundle,
        self_consistency_n=self_consistency_n,
        prompt_variants_map=prompt_variants_map,
        model=model_first,
        # 可传入自定义模型（若你的 FirstLayerDetector/Agent 支持）
        # 兼容性考虑：如果不支持该参数，请在 FirstLayerDetector 内部读取 config 或忽略
    )
    # 若你希望强制第一层用特定模型，可在 FirstLayerDetector 内部把 agent.model 替换为 model_first

    # 4) 初始化第二层（新版：四维度对比+Agent判别）
    aspect_weights = cfg.get("second_layer", {}).get("aspect_weights", {
        "常识冲突": 0.35, "夸张程度": 0.25, "偏见倾向": 0.20, "情绪煽动": 0.20
    })
    temperature_true = float(cfg.get("second_layer", {}).get("temperature_true", 0.2))
    temperature_fake = float(cfg.get("second_layer", {}).get("temperature_fake", 0.8))
    max_tokens_second = int(cfg.get("second_layer", {}).get("max_tokens", 800))

    second_layer = SecondLayerDetector(
        api_key=api_key,
        model=model_second,
        base_url=base_url,
        temperature_true=temperature_true,
        temperature_fake=temperature_fake,
        max_tokens=max_tokens_second,
        timeout=timeout,
        proxy=proxy,
        ca_bundle=ca_bundle,
        aspect_weights=aspect_weights
    )

    # 5) 报告器
    reporter = ReportGenerator(output_csv)

    cap = cfg.get('detection', {}).get('second_layer_trigger_fake_cap', 0.9)
    _, _, cap = thresholds(real_th, fake_th, cap)
    results = []
    for item in items:
        actual = _normalize_actual_label(item['label'])
        result = detect_item(item, first_layer, second_layer, cap)
        results.append(dict(result, sample_id=item['id'], title=item['title'], actual_label=actual))
    reporter.save_results(results)
    reporter.print_accuracy(results)
    if any(row['status'] == 'error' for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
