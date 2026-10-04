"""Shared training/inference device placement."""
def _normalize_agent_device_map(value, quantization: str):
    """
    让 config 里写 "cuda:0"/"cuda:1" 也能在量化模式下正常工作：
      - quantization in {"4bit","8bit"}:
          * "auto" -> "auto"
          * "cuda:N" -> {"": N}  # 把整个模型放到单卡
          * dict -> 原样
          * None -> None
      - quantization == "none":
          返回 None（不走 device_map），用 move_to_device 在后面 .to()
    """
    if quantization not in ("4bit", "8bit"):
        return None

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return "auto"
        import re
        m = re.match(r"cuda:(\d+)$", v)
        if m:
            return {"": int(m.group(1))}
        if v == "cpu":
            return {"": "cpu"}
        return None
    return None


def _normalize_move_to_device(value, quantization: str):
    if quantization in ("4bit", "8bit"):
        return None
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None
