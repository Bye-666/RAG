"""DashScope (阿里云千问) LLM 实现

支持的模型：
- qwen-max: 最强能力（默认）
- qwen-plus: 平衡性能与成本
- qwen-turbo: 快速响应
"""
import os
from typing import Any, Generator, Optional

from .base import BaseLLM


class DashScopeLLM(BaseLLM):
    """DashScope LLM 实现

    使用阿里云 DashScope API 调用千问系列模型。
    """

    def __init__(
        self,
        model: str = "qwen-max",
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """初始化 DashScope LLM

        Args:
            model: 模型名称，默认 qwen-max
            api_key: DashScope API Key，如果为 None 则从环境变量获取
            **kwargs: 其他参数（temperature, max_tokens 等）
        """
        super().__init__(model=model, api_key=api_key, **kwargs)

        # 获取 API Key
        if self.api_key is None:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DashScope API Key 未设置。请通过参数传入或设置环境变量 DASHSCOPE_API_KEY"
            )

        # 延迟导入，避免在未安装 dashscope 时导入失败
        try:
            import dashscope
            self.dashscope = dashscope
            self.dashscope.api_key = self.api_key
        except ImportError:
            raise ImportError(
                "dashscope 包未安装。请运行：pip install dashscope"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """同步生成文本

        Args:
            prompt: 用户输入提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 其他生成参数

        Returns:
            生成的文本内容

        Raises:
            Exception: API 调用失败时抛出异常
        """
        from dashscope import Generation

        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 合并参数
        generation_kwargs = {**self.kwargs, **kwargs}

        # 调用 API
        response = Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            **generation_kwargs
        )

        # 检查响应
        if response.status_code != 200:
            raise Exception(
                f"DashScope API 调用失败: {response.code} - {response.message}"
            )

        # 提取生成内容
        return response.output.choices[0].message.content

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """流式生成文本

        Args:
            prompt: 用户输入提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段（增量内容）

        Raises:
            Exception: API 调用失败时抛出异常
        """
        from dashscope import Generation

        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 合并参数，启用流式
        generation_kwargs = {**self.kwargs, **kwargs}
        generation_kwargs["stream"] = True

        # 调用 API（流式）
        responses = Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            **generation_kwargs
        )

        # 流式返回
        for response in responses:
            if response.status_code != 200:
                raise Exception(
                    f"DashScope API 调用失败: {response.code} - {response.message}"
                )

            # 提取增量内容
            if hasattr(response.output, "choices") and response.output.choices:
                delta = response.output.choices[0].message.get("content", "")
                if delta:
                    yield delta
