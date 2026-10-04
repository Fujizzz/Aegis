import logging
import json
import re
from typing import Optional, Dict, Any
from probability import parse_probability, normalize_comparison


def _build_http_client(timeout: float = 45.0, proxy: str = "", ca_bundle: str = ""):
    import httpx
    transport = httpx.HTTPTransport(retries=3)
    kwargs = {"transport": transport, "timeout": timeout}
    if proxy:
        kwargs["proxy"] = proxy
    if ca_bundle:
        kwargs["verify"] = ca_bundle
    return httpx.Client(**kwargs)


class DeepSeekAgent:
    """封装 DeepSeek API 调用，实现特定分析角色的 Agent（支持打分与对比判别）。"""

    def __init__(self, api_key: str, role: str, prompt_template: str,
                 model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com",
                 temperature: float = 0.2, max_tokens: int = 512,
                 timeout: float = 45.0, proxy: str = "", ca_bundle: str = "", client=None):
        self.api_key = api_key
        self.role = role
        self.prompt_template = prompt_template  # 模板: {title} {content} {subject} {date}
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if client is None:
            from openai import OpenAI
            http_client = _build_http_client(timeout=timeout, proxy=proxy, ca_bundle=ca_bundle)
            client = OpenAI(api_key=self.api_key, base_url=base_url, http_client=http_client)
        self.client = client

        self.logger = logging.getLogger(__name__)

    # ---------- 第一层用：单一维度 0~1 置信度 ----------
    def analyze(self, title: str, content: str, subject: str = "", date: str = "") -> float:
        prompt_text = self.prompt_template.format(
            title=title or "", content=content or "", subject=subject or "", date=date or ""
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": f"You are an expert news analysis agent specialized in {self.role}."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            content_text = resp.choices[0].message.content.strip()
            confidence = parse_probability(content_text)

            self.logger.info(f"Agent '{self.role}' confidence: {confidence:.3f}")
            return confidence

        except Exception as e:
            self.logger.exception("DeepSeek SDK call failed: %s", e)
            raise RuntimeError('First-layer API scoring failed') from e

    # ---------- 第二层用：比较原文与真/假稿在该维度的接近度 ----------
    def compare_closeness(self,
                          original: str,
                          true_text: str,
                          fake_text: str,
                          subject: str = "",
                          date: str = "",
                          compare_prompt_template: Optional[str] = None
                          ) -> Dict[str, Any]:
        """
        让LLM在本Agent维度上判断【原文】更接近【真稿】还是【假稿】。
        期望 JSON 输出: {"closer":"true|fake|tie","proximity_true":0~1,"proximity_fake":0~1}
        返回合法 dict；请求或解析失败时抛出异常，由流水线记录弃权状态。
        """
        # 默认对比模板（按角色定义评价准则）
        # default_compare = (
        #     "任务：仅在【{role}】维度上比较三段文本（原文/真实版/假版）。"
        #     "请在内部完成推理，**只输出一个JSON**，字段："
        #     # "`closer` ∈ {'true','fake','tie'}，"
        #     "`closer` 只能取值 'true'、'fake' 或 'tie'，"
        #     "`proximity_true` (0~1)，`proximity_fake` (0~1)，两者之和≈1。\n\n"
        #     "评价准则：\n"
        #     "- 情绪煽动：是否存在煽动/极化/耸动语气、过度感叹与夸张形容；\n"
        #     "- 夸张程度：绝对化词汇（如“完全/100%/史无前例/震惊”）、标题党、耸动标点；\n"
        #     "- 偏见倾向：单边叙事、选择性事实、对立面抹黑；\n"
        #     "- 常识冲突：与日常常识/已知事实明显相悖或夸大。\n\n"
        #     "原文：{original}\n\n"
        #     "真实版：{true_text}\n\n"
        #     "假版：{fake_text}\n"
        # )

        default_compare = (
            "Task: Compare three texts (Original / True version / Fake version) ONLY in the aspect of {role}.\n"
            "Think internally, and **output ONLY one JSON object** with the following fields:\n"
            "- `closer`: must be 'true', 'fake', or 'tie'\n"
            "- `proximity_true`: a value between 0 and 1\n"
            "- `proximity_fake`: a value between 0 and 1\n"
            "Note: proximity_true + proximity_fake ≈ 1.\n\n"
            "Evaluation criteria:\n"
            "- Emotion: Check if there is inflammatory, polarized, or sensational tone, overuse of exclamation or exaggeration.\n"
            "- Exaggeration: Presence of absolutist terms (e.g., 'completely', '100%', 'unprecedented', 'shocking'), clickbait, or sensational punctuation.\n"
            "- Bias: One-sided narrative, selective use of facts, or derogatory remarks against the opposite side.\n"
            "- Commonsense Conflict: Contradiction with daily commonsense or widely known facts, or obvious exaggeration.\n\n"
            "Original: {original}\n\n"
            "True version: {true_text}\n\n"
            "Fake version: {fake_text}\n"
        )

        prompt = (compare_prompt_template or default_compare).format(
            role=self.role, original=original, true_text=true_text, fake_text=fake_text,
            subject=subject or "", date=date or ""
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": f"You are an evaluation agent for the '{self.role}' dimension. "
                                f"Think step-by-step internally. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=max(256, self.max_tokens),
                stream=False
            )
            txt = resp.choices[0].message.content.strip()
            parsed = self._parse_compare_json(txt)
            if parsed is None:
                raise ValueError('Invalid comparison response')
            return parsed
        except Exception as e:
            self.logger.exception("DeepSeek SDK compare_closeness failed: %s", e)
            raise RuntimeError('Comparison API failed') from e

    # ---------- 解析辅助 ----------
    def _parse_confidence_from_text(self, text: str) -> float:
        return parse_probability(text)

    def _parse_compare_json(self, text: str) -> Optional[Dict[str, Any]]:
        # 尝试直接JSON
        try:
            data = json.loads(text)
            return self._normalize_compare_json(data)
        except Exception:
            pass
        # 从代码块中提取
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                return self._normalize_compare_json(data)
            except Exception:
                return None
        return None

    def _normalize_compare_json(self, data):
        return normalize_comparison(data)
