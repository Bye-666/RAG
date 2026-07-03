"""配置模块单元测试"""
import os
import tempfile
from pathlib import Path

import pytest

from src.config.settings import Settings, get_settings


class TestSettings:
    """Settings 类测试"""

    def test_load_config(self):
        """测试加载配置文件"""
        settings = get_settings()
        assert settings.config is not None
        assert isinstance(settings.config, dict)

    def test_get_simple_key(self):
        """测试获取简单配置键"""
        settings = get_settings()
        # 测试已存在的键
        assert settings.get("vector_store") is not None
        assert settings.get("vector_store.type") == "milvus"

    def test_get_nested_key(self):
        """测试获取嵌套配置键"""
        settings = get_settings()
        # 测试嵌套键
        vector_type = settings.get("vector_store.type")
        assert vector_type == "milvus"

    def test_get_default_value(self):
        """测试获取不存在的键返回默认值"""
        settings = get_settings()
        value = settings.get("nonexistent.key", "default_value")
        assert value == "default_value"

    def test_dict_access(self):
        """测试字典式访问"""
        settings = get_settings()
        vector_store = settings["vector_store"]
        assert vector_store is not None
        assert vector_store["type"] == "milvus"

    def test_contains(self):
        """测试 in 操作符"""
        settings = get_settings()
        assert "vector_store" in settings
        assert "nonexistent_key" not in settings

    def test_env_var_expansion(self):
        """测试环境变量替换"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
test_key: "${TEST_ENV_VAR}"
nested:
  api_key: "${TEST_API_KEY}"
normal_value: "no_expansion"
""")
            temp_config = f.name

        try:
            # 设置环境变量
            os.environ["TEST_ENV_VAR"] = "test_value"
            os.environ["TEST_API_KEY"] = "secret_key_123"

            # 加载配置
            settings = Settings(temp_config)

            # 验证环境变量被替换
            assert settings.get("test_key") == "test_value"
            assert settings.get("nested.api_key") == "secret_key_123"
            assert settings.get("normal_value") == "no_expansion"

        finally:
            # 清理
            os.unlink(temp_config)
            del os.environ["TEST_ENV_VAR"]
            del os.environ["TEST_API_KEY"]

    def test_env_var_not_exist(self):
        """测试环境变量不存在时保持原样"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
test_key: "${NONEXISTENT_VAR}"
""")
            temp_config = f.name

        try:
            settings = Settings(temp_config)
            # 环境变量不存在，保持原样
            assert settings.get("test_key") == "${NONEXISTENT_VAR}"
        finally:
            os.unlink(temp_config)

    def test_singleton_pattern(self):
        """测试单例模式"""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_file_not_found(self):
        """测试配置文件不存在时抛出异常"""
        with pytest.raises(FileNotFoundError):
            Settings("/nonexistent/path/settings.yaml")
