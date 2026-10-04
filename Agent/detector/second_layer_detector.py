import logging
from typing import Dict, List, Tuple, Optional
import os
from probability import normalized_weights, normalize_comparison, probability

from agent.deepseek_agent import DeepSeekAgent, _build_http_client


class SecondLayerDetector:
    """
    第二层检测器（重写版）：
    1) 用 LLM 生成「真稿」与「假稿」；
    2) 在四个维度（情绪煽动、夸张程度、偏见倾向、常识冲突）上，用对比型 Agent 判定
       原文更接近真稿/假稿，并输出 proximity_true/proximity_fake；
    3) 使用 margin→sigmoid 的方式，将各维度聚合为整体 p_fake，并应用“强假/强真优先”规则；
    4) 输出可解释说明、最终标签与 p_fake；类别置信度由流水线单独计算。
    """

    def __init__(self,
                 api_key: str,
                 model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 temperature_true: float = 0.2,
                 temperature_fake: float = 0.8,
                 max_tokens: int = 800,
                 timeout: float = 45.0,
                 proxy: str = "",
                 ca_bundle: str = "",
                 aspect_weights: Optional[Dict[str, float]] = None,
                 # 可选：语义相似度兜底（默认关闭，不联网）
                 use_embeddings: bool = False,
                 embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
                 embedding_cache: str = "",
                 hf_endpoint: str = "", client=None, compare_client_factory=None,
                 compare_prompt_template=None
                 ):
        self.logger = logging.getLogger(__name__)

        # 基本 LLM 客户端
        if client is None:
            from openai import OpenAI
            http_client = _build_http_client(timeout=timeout, proxy=proxy, ca_bundle=ca_bundle)
            client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.client = client
        self.compare_prompt_template = compare_prompt_template

        # 生成配置
        self.model = model
        self.temperature_true = float(temperature_true)
        self.temperature_fake = float(temperature_fake)
        self.max_tokens = int(max_tokens)

        # 维度设置（与第一层一致的四个维度）
        self.aspects: List[str] = ["情绪煽动", "夸张程度", "偏见倾向", "常识冲突"]
        self.aspect_weights: Dict[str, float] = ({
            "常识冲突": 0.35,
            "夸张程度": 0.25,
            "偏见倾向": 0.20,
            "情绪煽动": 0.20,
        } if aspect_weights is None else aspect_weights)
        self.aspect_weights = normalized_weights(self.aspect_weights, self.aspects)

        # 为每个维度准备一个对比型 Agent（其 prompt_template 在 compare_closeness 内部不直接使用）
        self.compare_agents: Dict[str, DeepSeekAgent] = {
            asp: DeepSeekAgent(
                api_key=api_key,
                role=asp,
                prompt_template="{content}",  # 占位
                model=model,
                base_url=base_url,
                temperature=0.2,
                max_tokens=512,
                timeout=timeout,
                proxy=proxy,
                ca_bundle=ca_bundle,
                client=compare_client_factory(asp) if compare_client_factory else None
            )
            for asp in self.aspects
        }

        # 可选：语义相似度兜底（默认不开）
        self.embedder = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                _st_available = True
            except ImportError:
                _st_available = False
            # 环境变量设置（镜像与加速）
            if hf_endpoint:
                os.environ["HF_ENDPOINT"] = hf_endpoint
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            if not _st_available:
                self.logger.warning("sentence-transformers not available, disable embedding fallback.")
            else:
                try:
                    if embedding_cache:
                        self.embedder = SentenceTransformer(embedding_model, cache_folder=embedding_cache)
                    else:
                        self.embedder = SentenceTransformer(embedding_model)
                    self.logger.info(f"Embedding fallback enabled with model: {embedding_model}")
                except Exception as e:
                    self.embedder = None
                    self.logger.warning(f"Embedder init failed. Fallback disabled. err={e}")

        self.logger.info(
            f"SecondLayer initialized. model={self.model}, temps(T/F)=({self.temperature_true}/{self.temperature_fake}), "
            f"max_tokens={self.max_tokens}, use_embeddings={bool(self.embedder)}"
        )

    # ===================== 对外主接口 =====================

    def analyze_news(self, title: str, content: str, subject: str = "", date: str = "") -> Tuple[str, str, float]:
        """
        输入新闻标题/正文/主题/日期，返回：
        (explanation: str, final_label: {'real'|'fake'}, p_fake: float[0~1])
        """
        # 1) 生成真/假稿
        true_article, fake_article = self._generate_true_fake(title, subject, date)

        # 2) 四维度对比（原文 vs 真/假）
        aspect_results: Dict[str, Dict] = {}
        for asp in self.aspects:
            res = self.compare_agents[asp].compare_closeness(
                original=content, true_text=true_article, fake_text=fake_article,
                subject=subject, date=date,
                compare_prompt_template=getattr(self, 'compare_prompt_template', None)
            )
            # 如需严谨，可在此进行自检：若closer与proximity大小矛盾，做一次交换或归一
            res = self._self_consistency_fix(res)
            aspect_results[asp] = res

        # 3) 聚合为整体假概率
        p_fake = self._aggregate_fake_probability(aspect_results, tau=0.15)

        # 4) 生成可解释说明 & 最终决策
        final_label = "fake" if p_fake >= 0.5 else "real"
        confidence = max(p_fake, 1 - p_fake)
        explanation = self._format_explanation(aspect_results, p_fake, final_label)

        self.logger.info(f"SecondLayer(4D) -> p_fake={p_fake:.3f}, final={final_label} ({confidence:.2f})")
        return explanation, final_label, probability(p_fake)

    # ===================== 生成模块 =====================

    def _generate_true_fake(self, title: str, subject: str, date: str) -> Tuple[str, str]:
        """
        使用 LLM 生成真稿与假稿。
        """
        # true_prompt = (
        #     "请根据给定信息撰写一篇客观、可核验的真实新闻稿，避免绝对化结论，可描述证据类型但不要编造具体机构/精确数据：\n"
        #     f"标题: {title}\n主题: {subject}\n日期: {date}\n"
        # )
        # fake_prompt = (
        #     "请根据给定信息撰写一篇夸张、带有误导性的假新闻稿，可以使用耸动措辞与绝对化断言，但避免引用真实机构名：\n"
        #     f"标题: {title}\n主题: {subject}\n日期: {date}\n"
        # )

        true_prompt = (
            "Please write a factual, objective, and verifiable news article based on the given information. "
            "Avoid absolute claims and exaggeration. You may describe the type of evidence but do not invent specific institutions or precise data.\n"
            f"Title: {title}\nSubject: {subject}\nDate: {date}\n"
        )
        fake_prompt = (
            "Please write a sensationalized, misleading, and exaggerated fake news article based on the given information. "
            "You may use emotional or exaggerated wording and absolute claims, but avoid citing real institution names or providing precise factual data.\n"
            f"Title: {title}\nSubject: {subject}\nDate: {date}\n"
        )

        try:
            resp_true = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You write factual, objective, verifiable news copy. Think internally, output only text."},
                    {"role": "user", "content": true_prompt}
                ],
                temperature=self.temperature_true,
                max_tokens=self.max_tokens,
                stream=False
            )
            resp_fake = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You write sensationalized, misleading fake news. Think internally, output only text."},
                    {"role": "user", "content": fake_prompt}
                ],
                temperature=self.temperature_fake,
                max_tokens=self.max_tokens,
                stream=False
            )
            true_article = resp_true.choices[0].message.content.strip()
            fake_article = resp_fake.choices[0].message.content.strip()
            if not true_article or not fake_article:
                raise ValueError('Empty generated reference')
            return true_article, fake_article

        except Exception as e:
            self.logger.exception("DeepSeek generation failed (SecondLayer): %s", e)
            # 失败必须向上报告，由流水线记录弃权。
            raise RuntimeError('Reference generation failed') from e

    # ===================== 聚合与解释 =====================

    def _aggregate_fake_probability(self, aspect_results: Dict[str, Dict], tau: float = 0.15) -> float:
        """
        使用 margin→sigmoid 聚合各维度假概率，并应用“强假/强真优先”规则。

        对每个维度：
          margin = (pf - pt) / tau
          p_fake_asp = sigmoid(margin)

        规则：
          - 若某维度 closer='fake' 且 pf>=0.75，则加强该维度的假概率贡献；
          - 若某维度 closer='true' 且 pt>=0.75，则压低该维度的假概率贡献；
          - 若出现强假维度/强真维度，整体再做一次轻微偏置（+/-0.05）。
        """
        import math

        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('tau must be finite and positive')
        if set(aspect_results) != set(self.aspects):
            raise ValueError('Incomplete aspect results')
        aspect_results = {k: normalize_comparison(v) for k, v in aspect_results.items()}
        self.aspect_weights = normalized_weights(self.aspect_weights, self.aspects)
        # 检查强信号
        strong_fake_hit = any(
            (str(r.get("closer", "")).lower() == "fake" and float(r.get("proximity_fake", 0.0)) >= 0.75)
            for r in aspect_results.values()
        )
        strong_true_hit = any(
            (str(r.get("closer", "")).lower() == "true" and float(r.get("proximity_true", 0.0)) >= 0.75)
            for r in aspect_results.values()
        )

        weighted_sum, weight_total = 0.0, 0.0
        for asp, r in aspect_results.items():
            pt = float(r.get("proximity_true", 0.5))
            pf = float(r.get("proximity_fake", 0.5))
            w = float(self.aspect_weights.get(asp, 1.0))

            margin = (pf - pt) / max(tau, 1e-6)
            p_fake_asp = 1.0 / (1.0 + math.exp(-max(-700, min(700, margin))))  # sigmoid

            closer = str(r.get("closer", "tie")).lower()
            if closer == "fake" and pf >= 0.75:
                p_fake_asp = min(0.95, 0.5 + 0.5 * p_fake_asp)  # 强调假信号
            elif closer == "true" and pt >= 0.75:
                p_fake_asp = max(0.05, 0.5 * p_fake_asp)       # 强调真信号

            weighted_sum += p_fake_asp * w
            weight_total += w

        if weight_total <= 0:
            return 0.5

        p_fake = weighted_sum / weight_total

        # 整体偏置
        if strong_fake_hit:
            p_fake = min(0.98, p_fake + 0.05)
        elif strong_true_hit:
            p_fake = max(0.02, p_fake - 0.05)

        return p_fake

    def _format_explanation(self, aspect_results: Dict[str, Dict], p_fake: float, final_label: str) -> str:
        parts = ["【第二层·四维度对比】"]
        for asp in self.aspects:
            r = aspect_results.get(asp, {})
            closer = str(r.get("closer", "tie"))
            pt = float(r.get("proximity_true", 0.5))
            pf = float(r.get("proximity_fake", 0.5))
            parts.append(f"- {asp}: closer={closer}, proximity_true={pt:.2f}, proximity_fake={pf:.2f}")
        parts.append(f"综合假概率 p_fake={p_fake:.2f} → 判定：{'假' if final_label=='fake' else '真'}")
        return " ".join(parts)

    # ===================== 解析/回退工具 =====================

    @staticmethod
    def _self_consistency_fix(res: Dict) -> Dict:
        """
        验证数值并归一化；closer 按归一化后的接近度确定。
        """
        return normalize_comparison(res)

    # ================ 可选的相似度兜底（仅当 embedder 存在时） ================

    def _semantic_similarity(self, a: str, b: str) -> float:
        if self.embedder is None:
            return self._simple_similarity(a, b)
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            va = self.embedder.encode(a)
            vb = self.embedder.encode(b)
            return float(cosine_similarity([va], [vb])[0][0])
        except Exception:
            return self._simple_similarity(a, b)

    @staticmethod
    def _simple_similarity(text1: str, text2: str) -> float:
        s1 = set(text1.split())
        s2 = set(text2.split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)
