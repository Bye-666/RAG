"""配置加载与管理

支持：
- YAML 配置文件加载
- 环境变量替换 ${VAR_NAME}
- 配置验证
- 单例模式访问
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Settings:
    """配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化配置

        Args:
            config_path: 配置文件路径，默认为 config/settings.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "settings.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        # 递归替换环境变量
        self._config = self._expand_env_vars(self._config)

    def _expand_env_vars(self, obj: Any) -> Any:
        """递归替换环境变量

        支持 ${VAR_NAME} 语法，如果环境变量不存在则保持原样

        Args:
            obj: 配置对象（字典、列表、字符串等）

        Returns:
            替换后的对象
        """
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # 匹配 ${VAR_NAME} 模式
            pattern = r"\$\{([^}]+)\}"

            def replacer(match: re.Match) -> str:
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))

            return re.sub(pattern, replacer, obj)
        else:
            return obj

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项

        支持点号分隔的嵌套键，例如 "llm.provider"

        Args:
            key: 配置键，支持点号分隔
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """字典式访问

        Args:
            key: 配置键

        Returns:
            配置值

        Raises:
            KeyError: 配置键不存在
        """
        value = self.get(key)
        if value is None and key not in self._config:
            raise KeyError(f"配置键不存在: {key}")
        return value

    def __contains__(self, key: str) -> bool:
        """检查配置键是否存在"""
        return self.get(key) is not None

    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return self._config

    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()


# 单例实例
_settings_instance: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None, force_reload: bool = False) -> Settings:
    """获取配置单例

    Args:
        config_path: 配置文件路径，仅首次调用时有效
        force_reload: 是否强制重新加载

    Returns:
        Settings 实例
    """
    global _settings_instance

    if _settings_instance is None or force_reload:
        _settings_instance = Settings(config_path)

    return _settings_instance
