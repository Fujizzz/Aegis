import logging
from typing import Dict, List, Tuple, Optional
import statistics

from agent.deepseek_agent import DeepSeekAgent
from probability import probability, normalized_weights, thresholds


class FirstLayerDetector:
    """
    第一层检测器：多 Agent 初筛。
    - 每个维度由一个 DeepSeekAgent 产生 0~1 的“可疑度/像假”分数；
    - 支持多样本自一致（self_consistency_n）与多 prompt 变体（prompt_variants_map）；
    - 将各维度分数按 weights 加权平均得到 combined_conf；
    - 与阈值比较后输出标签：'fake' | 'real' | 'uncertain'。
    """

    def __init__(self,
                 api_key: str,
                 aspects: Dict[str, str],
                 fake_threshold: float = 0.7,
                 real_threshold: float = 0.3,
                 base_url: str = "https://api.deepseek.com",
                 timeout: float = 45.0,
                 proxy: str = "",
                 ca_bundle: str = "",
                 # 新增能力：
                 weights: Optional[Dict[str, float]] = None,
                 self_consistency_n: int = 1,
                 prompt_variants_map: Optional[Dict[str, List[str]]] = None,
                 # 可选：覆盖默认模型名
                 model: str = "deepseek-chat",
                 temperature: float = 0.1,
                 max_tokens: int = 512, client_factory=None):
        """
        Parameters
        ----------
        api_key : DeepSeek API Key
        aspects : {维度名: prompt_template}，模板可使用 {title} {content} {subject} {date}
        fake_threshold : ≥ 该值判假
        real_threshold : ≤ 该值判真
        base_url/timeout/proxy/ca_bundle : 联网参数
        weights : {维度名: 权重}，未提供的维度默认 1.0
        self_consistency_n : 自一致次数（≥2 时效果更稳，成本更高）
        prompt_variants_map : {维度名: [模板1, 模板2, ...]}，供自一致轮换
        model/temperature/max_tokens : Agent 调用模型的参数
        """
        self.logger = logging.getLogger(__name__)

        self.real_threshold, self.fake_threshold, _ = thresholds(real_threshold, fake_threshold, 1.0)
        if not aspects:
            raise ValueError('At least one aspect is required')
        aliases = {'情绪煽动': 'Emotion/Sentiment', '夸张程度': 'Exaggeration',
                   '偏见倾向': 'Bias', '常识冲突': 'Commonsense Conflict'}
        selected = {name: 1.0 for name in aspects} if weights is None else weights
        mapped = {}
        for name, value in selected.items():
            key = name if name in aspects else aliases.get(name, name)
            if key in mapped:
                raise ValueError('Duplicate aspect weight aliases')
            mapped[key] = value
        self.weights = normalized_weights(mapped, aspects)
        if type(self_consistency_n) is not int or self_consistency_n < 1:
            raise ValueError('self_consistency_n must be a positive integer')
        self.self_consistency_n = self_consistency_n
        self.prompt_variants_map = prompt_variants_map or {}

        self.agents: Dict[str, DeepSeekAgent] = {}
        self.aspect_prompts: Dict[str, str] = {}

        # 为每个维度创建一个 Agent
        for aspect_name, prompt in aspects.items():
            agent = DeepSeekAgent(
                api_key=api_key,
                role=aspect_name,
                prompt_template=prompt,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                proxy=proxy,
                ca_bundle=ca_bundle,
                client=client_factory(aspect_name) if client_factory else None
            )
            self.agents[aspect_name] = agent
            self.aspect_prompts[aspect_name] = prompt

        self.logger.info(
            f"FirstLayer initialized with {len(self.agents)} aspects; "
            f"thresholds(real≤{self.real_threshold:.2f}, fake≥{self.fake_threshold:.2f}); "
            f"self_consistency_n={self.self_consistency_n}"
        )

    # ------------------------ 对外主接口 ------------------------
    def analyze_news(self,
                     title: str,
                     content: str,
                     subject: str = "",
                     date: str = "") -> Tuple[float, str]:
        """
        对一条新闻进行第一层分析。
        Returns
        -------
        (combined_conf, label)
          - combined_conf: 0~1，值越大越“像假”
          - label: 'fake' | 'real' | 'uncertain'
        """
        scores = []
        weights = []

        for aspect_name, agent in self.agents.items():
            # 该维度的多 prompt 变体（若有）
            variants = self.prompt_variants_map.get(aspect_name)
            # 打分（支持 analyze_multi，若无则回退多次 analyze）
            score = self._score_with_self_consistency(
                agent=agent,
                title=title,
                content=content,
                subject=subject,
                date=date,
                prompt_variants=variants,
                n=self.self_consistency_n
            )
            w = float(self.weights.get(aspect_name, 1.0))
            scores.append(score)
            weights.append(w)
            self.logger.info(f"[FirstLayer] {aspect_name}: score={score:.3f}, weight={w:.2f}")

        combined_conf = self._weighted_mean(scores, weights)

        # 阈值判决
        if combined_conf >= self.fake_threshold:
            label = "fake"
        elif combined_conf <= self.real_threshold:
            label = "real"
        else:
            label = "uncertain"

        self.logger.info(f"First layer combined: {combined_conf:.3f} -> {label}")
        return combined_conf, label

    # ------------------------ 内部工具 ------------------------
    def _score_with_self_consistency(self,
                                     agent: DeepSeekAgent,
                                     title: str,
                                     content: str,
                                     subject: str,
                                     date: str,
                                     prompt_variants: Optional[List[str]],
                                     n: int) -> float:
        """
        使用同一维度多次打分做自一致，优先调用 agent.analyze_multi；
        若该方法不存在，回退为循环 analyze 并取中位数。
        """
        if hasattr(agent, 'analyze_multi'):
            return probability(agent.analyze_multi(
                title=title, content=content, subject=subject, date=date,
                prompt_variants=prompt_variants, n=n))
        scores = []
        variants = prompt_variants or [self.aspect_prompts[agent.role]]
        for i in range(n):
            original = agent.prompt_template
            agent.prompt_template = variants[i % len(variants)]
            try:
                scores.append(probability(agent.analyze(title=title, content=content, subject=subject, date=date)))
            finally:
                agent.prompt_template = original
        return float(statistics.median(scores))

    @staticmethod
    def _weighted_mean(scores, weights):
        if not scores or len(scores) != len(weights):
            raise ValueError('Expected nonempty matching scores and weights')
        normalized = normalized_weights(dict(enumerate(weights)), range(len(scores)))
        return probability(sum(probability(s) * normalized[i] for i, s in enumerate(scores)))
