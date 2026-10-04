import logging

def init_logging(level: str = "INFO"):
    """初始化日志配置。"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # 返回根 logger（可选）
    logger = logging.getLogger()
    logger.info(f"Logging initialized at level: {level}")
    return logger
