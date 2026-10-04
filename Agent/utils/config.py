import yaml

def load_config(config_path: str):
    """从指定的 YAML 配置文件加载配置字典。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
