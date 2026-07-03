"""DashScope LLM 单元测试"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_dashscope():
    """Mock dashscope 模块"""
    mock_module = MagicMock()
    mock_module.Generation = MagicMock()
    with patch.dict("sys.modules", {"dashscope": mock_module}):
        yield mock_module


class TestDashScopeLLM:
    """DashScopeLLM 测试"""

    def test_import(self):
        """测试导入模块"""
        from src.libs.llm.dashscope_llm import DashScopeLLM
        assert DashScopeLLM is not None

    def test_init_with_api_key(self, mock_dashscope):
        """测试使用 API Key 初始化"""
        from src.libs.llm.dashscope_llm import DashScopeLLM
        llm = DashScopeLLM(api_key="test-key")
        assert llm.api_key == "test-key"
        assert llm.model == "qwen-max"

    def test_init_from_env(self, mock_dashscope):
        """测试从环境变量获取 API Key"""
        os.environ["DASHSCOPE_API_KEY"] = "env-test-key"
        try:
            from src.libs.llm.dashscope_llm import DashScopeLLM
            llm = DashScopeLLM()
            assert llm.api_key == "env-test-key"
        finally:
            del os.environ["DASHSCOPE_API_KEY"]

    def test_init_without_api_key(self, mock_dashscope):
        """测试未设置 API Key 时抛出异常"""
        # 确保环境变量不存在
        if "DASHSCOPE_API_KEY" in os.environ:
            del os.environ["DASHSCOPE_API_KEY"]

        from src.libs.llm.dashscope_llm import DashScopeLLM
        with pytest.raises(ValueError, match="API Key 未设置"):
            DashScopeLLM()

    def test_generate_success(self, mock_dashscope):
        """测试同步生成成功"""
        # Mock Generation.call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output.choices = [
            MagicMock(message=MagicMock(content="生成的文本内容"))
        ]
        mock_dashscope.Generation.call.return_value = mock_response

        from src.libs.llm.dashscope_llm import DashScopeLLM

        llm = DashScopeLLM(api_key="test-key")
        result = llm.generate("测试提示词")

        assert result == "生成的文本内容"
        mock_dashscope.Generation.call.assert_called_once()

    def test_generate_with_system_prompt(self, mock_dashscope):
        """测试带系统提示词的生成"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output.choices = [
            MagicMock(message=MagicMock(content="响应内容"))
        ]
        mock_dashscope.Generation.call.return_value = mock_response

        from src.libs.llm.dashscope_llm import DashScopeLLM

        llm = DashScopeLLM(api_key="test-key")
        result = llm.generate("用户提示词", system_prompt="系统提示词")

        assert result == "响应内容"
        # 验证消息列表包含系统和用户消息
        call_args = mock_dashscope.Generation.call.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_generate_api_error(self, mock_dashscope):
        """测试 API 调用失败"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.code = "InvalidParameter"
        mock_response.message = "参数错误"
        mock_dashscope.Generation.call.return_value = mock_response

        from src.libs.llm.dashscope_llm import DashScopeLLM

        llm = DashScopeLLM(api_key="test-key")

        with pytest.raises(Exception, match="API 调用失败"):
            llm.generate("测试")

    def test_generate_stream_success(self, mock_dashscope):
        """测试流式生成成功"""
        # Mock 流式响应
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.output.choices = [
            MagicMock(message=MagicMock(get=lambda k, d: "第一"))
        ]

        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.output.choices = [
            MagicMock(message=MagicMock(get=lambda k, d: "部分"))
        ]

        mock_dashscope.Generation.call.return_value = [mock_response1, mock_response2]

        from src.libs.llm.dashscope_llm import DashScopeLLM

        llm = DashScopeLLM(api_key="test-key")
        result = list(llm.generate_stream("测试提示词"))

        assert result == ["第一", "部分"]
        mock_dashscope.Generation.call.assert_called_once()

    def test_repr(self, mock_dashscope):
        """测试字符串表示"""
        from src.libs.llm.dashscope_llm import DashScopeLLM
        llm = DashScopeLLM(model="qwen-plus", api_key="test-key")
        assert "DashScopeLLM" in repr(llm)
        assert "qwen-plus" in repr(llm)

    def test_custom_model(self, mock_dashscope):
        """测试使用自定义模型"""
        from src.libs.llm.dashscope_llm import DashScopeLLM
        llm = DashScopeLLM(model="qwen-turbo", api_key="test-key")
        assert llm.model == "qwen-turbo"

    def test_kwargs_passed(self, mock_dashscope):
        """测试额外参数传递"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output.choices = [
            MagicMock(message=MagicMock(content="测试"))
        ]
        mock_dashscope.Generation.call.return_value = mock_response

        from src.libs.llm.dashscope_llm import DashScopeLLM

        llm = DashScopeLLM(api_key="test-key", temperature=0.7, max_tokens=100)
        llm.generate("测试")

        # 验证参数传递
        call_args = mock_dashscope.Generation.call.call_args
        assert "temperature" in call_args.kwargs or llm.kwargs.get("temperature") == 0.7
